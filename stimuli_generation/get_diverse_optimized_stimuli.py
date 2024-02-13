"""
Code for generating optimized stimuli, i.e. feature visualizations.
The ideal diversity term is found using a binary search over a reasonable interval.
"""

import os
import time
import logging
import random
from datetime import datetime as dt
from argparse import ArgumentParser

from typing import Tuple, List, Sequence, Callable
import torch
from torchvision.transforms import Normalize
from lucent.optvis import render, param, transform, objectives
import numpy as np
import imageio

from stimuli_generation.utils import (
    load_model,
    split_unit,
    read_units_file,
    get_activations,
    get_min_max_exemplar_activations,
    STORAGE_DIR,
    store_units,
    accuracies,
    test_permute,
)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class RGBGradientNormInterruptCheck:
    """Interrupts the rendering once the gradient norm becomes flat.

    Args:
        min_steps: Minimum number of steps to take before checking for convergence.
        window_length: Number of gradients to consider for checking for flatness.
        target_ratio: The ratio of the gradient norm in the first half of the window
            to the gradient norm in the second half of the window. If the ratio is
            below this value, the gradient norm is considered flat.
        smoothing_weight: The weight of the previous gradient norm when calculating
            the exponentially smoothed gradient norm.
    """

    def __init__(self, min_steps: int = 2500,
                 window_length: int = 2000,
                 target_ratio: float = 1.1,
                 smoothing_weight: float = 0.99):
        self.min_steps = min_steps
        self.window_length = window_length
        self.target_ratio = target_ratio
        self.smoothing_weight = smoothing_weight
        self._smooth_grad_norms = []

    @staticmethod
    @torch.no_grad()
    def _get_mean_rgb_grad(fft_grads: torch.Tensor, image_shape: Sequence[int]):
        if type(fft_grads) is not torch.complex64:
            fft_grads = torch.view_as_complex(fft_grads)

        if len(image_shape) != 2:
            raise ValueError(
                "image_shape must be a 2-tuple of the image's spatial dimensions.")
        h, w = image_shape
        # NOTE in lucent.param.spatial.fft_image(), they scale the spectrum first.
        ifft = torch.fft.irfftn(fft_grads, s=(h, w), norm='ortho')

        # Look at the max of the batch, to make sure that even the worst image is okay.
        return torch.norm(ifft, p=2, dim=(1, 2, 3)).max().detach().cpu().numpy()

    def reset(self):
        """Resets the state of the interrupt check."""
        self._smooth_grad_norms = []

    def step(self, images_param_grad: torch.Tensor, image_shape: Sequence[int]):
        """Checks if the gradient has become flat, i.e., its norm has vanished.

        Args:
            images_param_grad: The gradient of the image parameter.
            image_shape: The shape of the image.
        """
        mean_rgb_grad = RGBGradientNormInterruptCheck._get_mean_rgb_grad(
            images_param_grad, image_shape)
        # Calculate exponentially smoothed value to stabilize stopping criterion.
        last = self._smooth_grad_norms[-1] if self._smooth_grad_norms else mean_rgb_grad
        smoothed_val = last * self.smoothing_weight + (
                    1 - self.smoothing_weight) * mean_rgb_grad
        self._smooth_grad_norms.append(smoothed_val)

        if len(self._smooth_grad_norms) > max(self.window_length, self.min_steps):
            first_half = np.mean(
                self._smooth_grad_norms[-self.window_length:-self.window_length // 2])
            second_half = np.mean(self._smooth_grad_norms[-self.window_length // 2:])
            if first_half / second_half <= self.target_ratio:
                raise render.RenderInterrupt()


def create_base_dir(args):
    """
    Creates the target directory in which to store optimized stimuli.

    :param args: the CLI arguments
    """
    base_dir = os.path.join(STORAGE_DIR, "stimuli", args.model_name)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def obtain_fvs(maximise, args, model, unit, diversity_alpha, num_images):
    """
    Creates the visualizations for a specified unit.

    :param maximise: whether to obtain positive activations (True) or negative ones
        (False)
    :param args: the CLI arguments
    :param model: the torch model
    :param unit: the unit as a string
    :param diversity_alpha: the diversity term to be used
    :param num_images: how many images to generate

    :returns: list of tuples of (img, activation)
    """

    # Resetting random seeds for generation attempt, to make the results for individual units as reproducible as possible.
    np.random.seed(args.seed)
    # Just setting random seed in case any other dependency uses random somewhere
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    layer, channel = split_unit(unit)

    if "clip" in args.model_name:
        layer = layer[len("visual_"):]

    logging.info(
        f"Visualizing unit {unit} at {str(dt.now())} with diversity {diversity_alpha}"
    )
    start_time = int(time.time())

    # Pad visualization to 256x256 (we'll crop it later) and batch visualization
    # See https://distill.pub/2017/feature-visualization/
    img_size = 224
    padding_size = 16
    padded_img_size = img_size + 2 * padding_size

    def batch_param_f():
        return param.image(padded_img_size, batch=num_images)

    # Define objectives
    obj = objectives.channel(
        layer,
        channel,
        channel_mode="last" if test_permute(args.model_name, layer) else "first"
    )

    # Obtain negatively activating images
    if not maximise:
        obj = -1 * obj

    # Add diversity term
    if diversity_alpha != 0:
        obj -= diversity_alpha * objectives.diversity(
            layer,
            channel_mode="last" if test_permute(args.model_name, layer) else "first")

    # center-crop transform for clip-resnet50, which doesn't like larger images
    def center_crop(h: int, w: int) -> Callable[[torch.Tensor], torch.Tensor]:
        def inner(x: torch.Tensor) -> torch.Tensor:
            assert x.shape[2] >= h
            assert x.shape[3] >= w

            oy = (x.shape[2] - h) // 2
            ox = (x.shape[3] - w) // 2

            return x[:, :, oy:oy+h, ox:ox+w]

        return inner

    # Define the transforms and optimizer
    preprocess = True  # whether normal transforms should be applied within render
    transforms = [
        transform.jitter(16),
        transform.random_scale((1.0, 0.975, 1.025, 0.95, 1.05)),
        transform.random_rotate((-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)),
        transform.jitter(8),
    ]

    if args.model_name in ("clip-resnet50", "in21k_in1k-vit_b32", "clip-vit_b32"):
        preprocess = False  # use custom transforms, not default
        custom_transforms = [
            center_crop(224, 224),  # no padding for clip resnet
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ]
        transforms.extend(custom_transforms)

    def get_optimizer(params):
        return torch.optim.Adam(params, 0.05)

    checker = RGBGradientNormInterruptCheck()

    def iteration_callback(hook, loss, images, images_params):
        checker.step(images_params[0].grad, (padded_img_size, padded_img_size))

    # Generate max stimuli
    images = render.render_vis(
        model,
        obj,
        param_f=batch_param_f,
        thresholds=(20000,),  # Hard cutoff at 20000 steps, never doing more than that
        verbose=True,
        transforms=transforms,
        optimizer_f=get_optimizer,
        show_image=False,
        progress=False,
        preprocess=preprocess,
        iteration_callback=iteration_callback,
    )[-1]

    # Crop images back to 224x224
    def crop_and_list(images):
        images = images[:, padding_size:-padding_size, padding_size:-padding_size, :]
        return [images[i, ...] for i in range(images.shape[0])]

    images = crop_and_list(images)

    # Get activations actually achieved by these FVs
    acts = get_activations(model, args.model_name, layer, channel, images)

    # Sort both lists, according to activations, and select strongest num_images
    # NOTE: max is sorted increasing, min decreasing, so that in both lists, 9 comes
    # last (because from least to most activating, we go from min_9 to max_9).
    images = list(
        sorted(zip(images, acts), key=lambda pair: pair[1], reverse=not maximise)
    )

    end_time = int(time.time())
    logging.info(
        f"Generating FVs for unit {unit} at diversity {diversity_alpha} took "
        f"{(end_time-start_time)/60:.2f} minutes."
    )

    return images


def store_images(base_dir, layer, channel, min_results, max_results):
    """
    Takes lists of (img, act)-pairs and stores them.

    :param base_dir: the base directory in which to create unit-directory
    :param layer: the layer of the unit
    :param channel: the channel of the unit
    :param min_results: minimally activating FVs, list of (img, act)
    :param max_results: maximally activating FVs, list of (img, act)
    """
    unit_dir = os.path.join(base_dir, layer, f"channel_{channel}", "optimized_images")
    os.makedirs(unit_dir, exist_ok=True)

    def store_images(name, results):

        images = [img for img, _act in results]

        for i, img in enumerate(images):
            save_name = os.path.join(unit_dir, f"{name}_{i}.png")
            imageio.imwrite(save_name, np.uint8(img * 255))

    store_images("min", min_results)
    store_images("max", max_results)


def visualize_unit(args, model, unit, base_dir):
    """
    Creates and stores the visualizations for a specified unit.
    Finds the largest possible diversity term that still exceeds the activation of
    natural exemplars.

    :param args: the CLI arguments
    :param model: the torch-model
    :param unit: the unit as a string
    :param base_dir: the base directory for output

    :returns: A boolean (True if FVs were successfully written, False otherwise) and a
    tuple of min/max diversities
    """

    # From unit name, get the layer and channel
    layer, channel = split_unit(unit)

    # Get the min / max activations achieved by exemplars
    min_exemplar_act, max_exemplar_act = get_min_max_exemplar_activations(
        args.model_name,
        unit,
        args.csv
    )
    logging.info(
        f"Unit {unit} achieved exemplars with min {min_exemplar_act:.3f} and "
        f"max {max_exemplar_act:.3f}."
    )

    # Get the min / max activations achieved by feature visualization without diversity
    min_fvs_no_div = obtain_fvs(False, args, model, unit, 0, args.num_images)
    max_fvs_no_div = obtain_fvs(True, args, model, unit, 0, args.num_images)

    # When selecting the max / min nondiverse activation, select the weakest FV of
    # the batch, to be consistent with how the diverse batches are treated:
    # There, we also select the weakest image of the batch.
    min_fv_act = max(act for _, act in min_fvs_no_div)
    max_fv_act = min(act for _, act in max_fvs_no_div)
    logging.info(
        f"Unit {unit} achieved non-diverse FVs with min {min_fv_act:.3f} and max "
        f"{max_fv_act:.3f}."
    )

    # If exemplars were more strongly activating than even nondiverse FVs, the
    # optimization procedure did not work properly, so we abort

    min_fv_too_weak = min_fv_act > min_exemplar_act
    max_fv_too_weak = max_fv_act < max_exemplar_act
    success = not (min_fv_too_weak or max_fv_too_weak)

    if min_fv_too_weak:
        logging.error(
            f"Min-Exemplar was more negatively activating for unit {unit} than "
            "Min-nondiverse-FV!"
        )

    if max_fv_too_weak:
        logging.error(
            f"Max-Exemplar was more positively activating for unit {unit} than "
            "Max-nondiverse-FV!"
        )

    if min_fv_too_weak and max_fv_too_weak:
        logging.error(
            f"Both Min- and Max-Exemplars were too weakly activating for unit {unit}!"
        )
        # still storing nondiverse images, but returning False so unit gets added to skipped list 
        store_images(base_dir, layer, channel, min_fvs_no_div, max_fvs_no_div)
        return False, (-1, -1)

    # The target activation is the activation value our FVs should achieve at least.
    # We set this to exemplar activation, then push the diversity as high as possible.
    min_target_act = min_exemplar_act
    max_target_act = max_exemplar_act

    # I hope this never happens, because this means something went wrong in the
    # activation collection
    assert max_target_act > min_target_act, (
        f"Minimally activating FV was larger than maximally activating FV for {unit}! "
        f"Skipping this unit!"
    )

    # If we don't care about diversity: Just store these images and be done with it.
    if args.no_diversity:
        store_images(base_dir, layer, channel, min_fvs_no_div, max_fvs_no_div)
        logging.info(f"{'SUCCESS' if success else 'FAILURE'} for unit {unit}!")
        return success, (0, 0)

    # Helper functions to extract the weakest FVs and compare them against the
    # exemplars.
    def check_weakest_negative(fvs):
        weakest_act = fvs[0][1]
        logging.info(f"Weakest negative FV achieved {weakest_act:.3f}")
        return weakest_act < min_target_act

    def check_weakest_positive(fvs):
        weakest_act = fvs[0][1]
        logging.info(f"Weakest positive FV achieved {weakest_act:.3f}")
        return weakest_act > max_target_act

    def exponential_search(
        maximise: bool, lower_bound: float = 1.0, upper_bound: float = 999_999
    ) -> Tuple[float, List[Tuple[np.ndarray, float]]]:
        """
        Performs an exponential search for the largest diversity term that still
        exceeds the target activation.
        For maximizing FVs, returns the *lowest* value that didn't work anymore,
        i.e. the upper bound of the following binary search.

        :param lower_bound: smallest value for the diversity term that should be
            checked in exponential search
        :param maximise: generate maximally activating if True, minimally otherwise
        :param upper_bound: biggest value of the diversity value that should be
            attempted, i.e. we do 100K but not 1M

        :returns: tuple(bound, imgs_and_acts), where bound is the upper bound of the
            diversity term and
            imgs_and_acts is a list of FVs and their activations for the biggest
            diversity-value that still worked, i.e. the search might
            return (1000, <imgs_with_div=100>)
        """

        logging.info(f"Running exponential search for unit {unit}.")

        check = check_weakest_positive if maximise else check_weakest_negative
        last_working_images = None
        div_alpha = lower_bound

        while div_alpha <= upper_bound:
            images_and_acts = obtain_fvs(
                maximise, args, model, unit, div_alpha, args.num_images - 1
            )  # Only generating args.num_images - 1 because last one is non-diverse

            if check(images_and_acts):
                logging.info(
                    f"Unit {unit} achieved min-activation of "
                    f"{images_and_acts[0][1]:.3f} for div-alpha {div_alpha},"
                    "increasing 10x."
                )
                last_working_images = images_and_acts
                div_alpha *= 10
            else:
                # Minimum exceeded, this div-alpha was too high = upper bound for
                # binary search
                logging.info(
                    f"Unit {unit} achieved min-activation of "
                    f"{images_and_acts[0][1]:.3f} for div-alpha {div_alpha}, returning."
                )

                return div_alpha, last_working_images

        logging.warning(f"Exponential search limit was reached for unit {unit}!")
        return div_alpha, last_working_images

    def binary_search(
        left_diversity: float, right_diversity: float, maximise: bool
    ) -> Tuple[float, List[Tuple[np.ndarray, float]]]:
        """
        Performs a binary search over the ideal diversity term for feature vis.:
        Highest possible value that still produces FVs that exceed best exemplar.

        :param left_diversity: left boundary of search interval
        :param right_diversity: right boundary of search interval
        :param maximise: generate maximally activating if True, minimally otherwise

        :returns: tuple(ideal_diversity_term, imgs_and_acts), where
            ideal_diversity_term is the best diversity term that could be found and
            imgs_and_acts is a list of FVs and their activations for this diversity term
        """

        assert (
            right_diversity > left_diversity
        ), "ERROR: Invalid interval for binary search!"

        check = check_weakest_positive if maximise else check_weakest_negative

        # Placeholder for current best min or max FVs and the diversity with which
        # they were found
        current_best_images = None
        current_diversity = None

        # Perform binary search over bounds, do not more than args.max_loops loops
        for loops in range(args.max_loops):

            middle = (left_diversity + right_diversity) / 2
            fvs = obtain_fvs(
                maximise, args, model, unit, middle, args.num_images - 1
            )  # Only generating num_images-1 because last one is non-diverse

            # If this worked, we go higher and try to find an even larger diversity term
            if check(fvs):
                current_best_images = fvs.copy()
                current_diversity = middle
                left_diversity = middle
            else:
                # If this did not work, we look at the left side
                right_diversity = middle

        return current_diversity, current_best_images

    def search(maximise: bool) -> Tuple[float, list]:
        """
        Performs an exponential search followed by a binary search to find biggest
        possible diversity term.

        :param maximise: generate maximally activating if True, minimally otherwise

        :returns: tuple (diversity, images) where diversity is the best diversity term
            that could be found and
            images is the list of image-activation-pairs that was generated
        """

        exp_search_lower_bound = 1
        exp_search_upper_bound = 999_999

        # If exponential search for div = 1 did not produce sufficiently activating FVs,
        # returns (1, None), otherwise (div, img-batch)
        exponential_diversity, exponential_images = exponential_search(
            maximise, exp_search_lower_bound, exp_search_upper_bound
        )

        # Exponential search never found an upper bound, so use the biggest value that
        # was tested
        if exponential_diversity >= exp_search_upper_bound:
            return exponential_diversity / 10, exponential_images

        # Running binary search next, starting either at 0 or at last value that worked
        # in exponential search
        starting_point = (
            0
            if exponential_diversity == exp_search_lower_bound
            else exponential_diversity / 10
        )
        binary_diversity, binary_images = binary_search(
            starting_point, exponential_diversity, maximise
        )

        # Maybe the binary search didn't find anything, then use result of exponential
        # search, unless that also didn't work, then return nondiverse and warn
        if binary_images is None:
            if exponential_images is None:
                logging.warning(
                    f"Search for unit {unit} did not work! Returning non-diverse FVs "
                    "with activation {0} instead!".format(
                        np.round(max_fv_act if maximise else min_fv_act, 3)
                    )
                )
                best_images = max_fvs_no_div if maximise else min_fvs_no_div
                return 0, best_images
            else:
                return exponential_diversity / 10, exponential_images

        # Binary search worked like intended, return result
        return binary_diversity, binary_images

    # Actually performing the searches.

    # Generate negatively activating images
    neg_div, min_images = search(False)

    logging.info(
        f"Completed search for negatively activating images for unit {unit}, "
        f"final diversity: {neg_div}!"
    )

    # Generate positively activating images
    pos_div, max_images = search(True)

    logging.info(
        f"Completed search for positively activating images for unit {unit}, "
        f"final diversity: {pos_div}!"
    )

    # Report failure if either of the two did not work
    if max_images is None or min_images is None:
        logging.error(
            f"Could not generate sufficiently activating images for unit {unit}! "
            f"Skipping!"
        )
        return False, (neg_div, pos_div)

    # Otherwise: Create target directory and save images

    # Append one non-diverse FV at the end of both lists.
    # Both lists are sorted from weakest to strongest, and the non-diverse FVs are as
    # well, so we can just append their last one as the overall last one.
    min_images.append(min_fvs_no_div[-1])
    max_images.append(max_fvs_no_div[-1])
    store_images(base_dir, layer, channel, min_images, max_images)

    logging.info(f"{'SUCCESS' if success else 'FAILURE'} for unit {unit}!")
    return success, (neg_div, pos_div)


def main(args):
    """
    Generate feature visualizations.
    The output paths will look something like:
    ./stimuli/{model_name}/{layer_name}/{unit_name}/optimized_images/{min/max}_{i}.png

    :param args: the CLI arguments
    """

    # Setting torch to deterministic mode.
    # Full reproducibility is not possible at the moment, since avg_pool3d_backward_cuda does not have a deterministic implementation.
    # See https://github.com/pytorch/pytorch/issues/72766 for progress on this.
    # In use_deterministic_algorithms, one could set warn_only=True, but this noticeably increases execution time.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)

    logging.basicConfig(
        format="%(levelname)s:  %(message)s",
        level=logging.INFO,  # Don't log DEBUG, but INFO, WARNING, ERROR and CRITICAL
    )

    base_dir = create_base_dir(args)

    model = load_model(args.model_name)

    if "clip" in args.model_name:
        model = model.visual

    # Visualize all units and count how many had to be skipped
    skipped_units = []  # Units for which the procedure failed
    diversities = {}
    for unit in args.units:

        try:

            # Returns whether a solution was found and what the diversities were
            success, (neg_div, pos_div) = visualize_unit(args, model, unit, base_dir)
            diversities[unit] = (neg_div, pos_div)

            # Note which units were skipped
            if not success:
                skipped_units.append(unit)

        except:  # noqa: E722
            # Catching blank exception on purpose
            # If anything unforeseen happens, I only want to lose this unit and not
            # the next ~40 hours of GPU time - just print the error and do the next one
            logging.error(f"Fatal problem with unit {unit}!", exc_info=True)
            skipped_units.append(unit)

    logging.info(f"Done! Skipped {len(skipped_units)} of {len(args.units)} units.")

    # Record failed units
    if skipped_units:
        store_units(skipped_units, args.skipped_units_file)

    # Record the actual diversities that were used for the units
    diversity_file = os.path.join(
        os.path.dirname(args.skipped_units_file), f"diversity_{args.gpu}.csv"
    )

    # Write all diversity terms
    with open(diversity_file, "w") as file:
        file.write("unit,negative_div,positive_div\n")  # Write CSV header
        for unit, (neg_div, pos_div) in diversities.items():
            file.write(",".join([str(x) for x in [unit, neg_div, pos_div]]) + "\n")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        choices=list(accuracies.keys()),
        required=True,
        help="Which model to use. Supported values are: "
             f"{', '.join(list(accuracies.keys()))}.",
    )

    parser.add_argument(
        "--units_file",
        type=str,
        required=True,
        help="Path to json-file with unit names as generated by sample_units.py.",
    )

    parser.add_argument(
        "--max_loops",
        type=int,
        default=7,
        help="How many loops the binary search should do.",
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=9,
        help="How many images to generate per unit. This includes one non-diverse FV "
        "when generating diverse FVs.",
    )

    parser.add_argument(
        "--skipped_units_file",
        type=str,
        required=True,
        help="File to which we store units that had to be skipped.",
    )

    parser.add_argument(
        "--no_diversity",
        action="store_true",
        help="Whether to generate FVs without any diversity.",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID that this job runs on.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator.",
    )

    parser.add_argument(
        "--csv",
        action="store_true",
        help="If the old method of reading activations from CSV should be used.",
    )

    arguments = parser.parse_args()

    # Read units from file
    arguments.units = read_units_file(arguments.units_file)

    main(arguments)
