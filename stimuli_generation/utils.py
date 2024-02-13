"""
Utility classes and methods.
"""

import json
import os
import re
import pickle
import platform
from typing import Callable, OrderedDict, Optional

import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import torchvision
from lucent.modelzoo import inceptionv1
from open_clip import create_model_and_transforms
from lucent.optvis.render import ModelHook
from PIL import Image, ImageFile
from torchvision import datasets, transforms
import timm

from stimuli_generation.clip_utils import zero_shot_classifier, imagenet_classnames, openai_imagenet_template
from stimuli_generation import resnet


# getting around PIL errors on Thomas' machine only - only relevant for debugging stuff
if platform.system() == "Darwin":
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {DEVICE}")

    # it's actually important that this ends with /
    IMAGENET_PATH = "/Users/thomas/tinyImageNet/"

    # specify where to save checkpoints
    os.environ["TORCH_HOME"] = "tmp/"

    # where to save data
    STORAGE_DIR = "tmp/test/"

else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {DEVICE}")

    # it's actually important that this ends with /
    IMAGENET_PATH = "/imagenet/"

    # specify where to save checkpoints
    os.environ["TORCH_HOME"] = "/torch_models/"

    # where to save data
    STORAGE_DIR = "/output/"

ckpt_dict = {
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet50-linf": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps4.0.ckpt",  # noqa: E501
    "resnet50-l2": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps3.ckpt",  # noqa: E501
    "wide_resnet50": "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
}

# Storing the advertised accuracies of the models, so we can check whether they
# work correctly
accuracies = {
    "resnet50": 0.7613,
    "resnet50-linf": 0.6386,
    "resnet50-l2": 0.6238,
    "googlenet": 0.6915,
    "clip-resnet50": 0.5983,
    "wide_resnet50": 0.8160,  # Using 232 instead of 256 resize before central crop
    "densenet_201": 0.7689,
    "convnext_b": 0.838,
    "clip-vit_b32": 0.666,
    "in1k-vit_b32": 0.74904
}

# Holds the transforms applied for each model
model_transforms = {
    model_name: None for model_name in accuracies
}

# Holds the necessary transforms for each model, so we don't have to construct them every time.
# For CLIP-models, getting the transforms requires loading the model, so it's a lot faster to just store them.
# This dict will be filled by the first call to get_transforms and used in consecutive calls.
model_transforms = {
    model_name: None for model_name in accuracies
}

def split_unit(layer_unit):
    """
    Splits a fully described unit into layer and index.

    :param layer_unit: the full unit, e.g. "layer1_0_conv1__0"
    """
    parts = layer_unit.split("__")
    return parts[0], int(parts[1])


def chunks(lst, n):
    """
    Yields successive n-sized chunks from lst.

    :param lst: the list to be chunked
    :param n: the size of each chunk
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def store_units(units, filename):
    """
    Stores a json-file of unit names in the given file.

    :param units: list of units
    :param filename: name of the file to store list in (without extension)
    """
    data = {"units": units}
    with open(filename + ".json", "w", encoding="utf-8") as file:
        json.dump(data, file)


def test_layer_relevance(layer, model_name):
    """
    Returns true if a layer-name is relevant, for a given model_name.

    :param layer: the layer
    :param model_name: the model
    """

    if model_name == "googlenet":
        return layer.startswith("mixed") and "pre_relu_conv" in layer

    if model_name == "densenet_201":
        dense = layer.startswith("features_denseblock") and layer.endswith(
            ("conv1", "conv2", "norm1", "norm2"))
        transition = layer.startswith("features_transition") and layer.endswith(("conv", "norm"))
        return dense or transition

    if model_name == "clip-vit_b32":
        return layer.startswith("transformer_resblocks") and layer.endswith(
            ("ln_1", "ls_1", "ln_2", "mlp_c_fc", "mlp_c_proj", "ls_2")
        )

    if model_name == "in1k-vit_b32":
        return layer.startswith("blocks_") and layer.endswith(
            ("norm1", "ls1", "norm2", "mlp_fc1", "mlp_fc2", "ls2")
        )

    if model_name == "convnext_b":
        regex = "stages_\\d+_blocks_\\d+"
        block = re.fullmatch(regex, layer) is not None  # corresponds to shortcut connection
        other = re.match(regex, layer) is not None and layer.endswith(
            ("conv_dw", "norm", "mlp_fc1", "mlp_fc2"))

        return block or other

    return (layer.startswith("layer") or layer.startswith("visual_layer")) and layer.endswith(
        ("conv1", "conv2", "conv3", "bn1", "bn2", "bn3", "shortcut"))


def test_permute(model_name, layer_name):
    """
    Tests if the activations at a layer should be permuted from (batch, h, w, c) to (batch, c, h, w).
    This is the case for 1x1-Conv- and Norm-layers in ConvNext,
    because they use LayerNorm and implement 1x1 convolutions as LinearLayers.

    :param model_name: the name of the model
    :param layer_name: the name of the layer
    :returns: True if the layer needs to be permuted
    """

    return model_name == "convnext_b" and ("mlp_fc" in layer_name or "norm" in layer_name)


def get_relevant_layers(model, model_name):
    """
    Returns the layers of interest to us as a list.

    :param model: the torch model
    :param model_name: str, model name
    :returns: list of str, layers
    """
    return [
        layer
        for layer in get_model_layers(model, False)
        if test_layer_relevance(layer, model_name)
    ]


def get_model_layers(model, get_modules=False):
    """
    Custom version of Lucent's modelzoo.util.get_model_layers that returns dict mapping
    layer name to layer.
    If get_modules is True, return a OrderedDict of layer names, layer representation
    string pair.
    """
    layers = OrderedDict() if get_modules else []

    # recursive function to get layers
    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                if get_modules:
                    layers["_".join(prefix + [name])] = layer
                else:
                    layers.append("_".join(prefix + [name]))
                get_layers(layer, prefix=prefix + [name])

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    get_layers(model)
    return layers


def read_units_file(unitfile):
    """
    Reads the json file which contains the list of units.
    (Overkill to have a function here but maybe this gets more complicated)

    :param unitfile: the json-file with units
    """
    with open(unitfile, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["units"]


def get_layers_from_units_list(units):
    """
    Takes a list of units and returns a list of all of their layers.

    :param units: list[str] of units
    :returns: list[str] of layers
    """
    layers = []
    for unit in units:
        layer, _ = split_unit(unit)
        layers.append(layer)

    return sorted(list(set(layers)))


def transform_and_copy_img(src_path, dest_path):
    """
    apply a transform to image at src_path and save it in dest_path
    """

    resize_and_crop = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )

    img = Image.open(src_path)
    img = resize_and_crop(img)

    # there were some issues with images being in CMYK mode
    if img.mode != "RGB":
        print(
            f"image {src_path} is not in RGB mode! Converting before "
            f"saving to {dest_path}"
        )
        img = img.convert("RGB")
    img.save(dest_path)


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths.
    Extends torchvision.datasets.ImageFolder.
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


def get_transforms(model_name):
    """
    Obtains a Composition of the model-specific transforms.
    :param model_name: the identifier of the model
    :returns: a torch transform
    """

    # check if we have transforms, use them if so, otherwise build them
    if model_transforms[model_name] is not None:
        return model_transforms[model_name]

    if model_name == "clip-resnet50":
        # openclip can conveniently just give us the transforms
        _, _, transformations = create_model_and_transforms('RN50', pretrained='openai')
    elif model_name == "clip-vit_b32":
        _, _, transformations = create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
    elif model_name == "in1k-vit_b32":
        model = timm.create_model("vit_base_patch32_224.augreg_in1k", pretrained=True)
        data_config = timm.data.resolve_data_config(args={}, model=model)
        transformations = timm.data.create_transform(
            **data_config, is_training=False)
    else:

        if model_name == "googlenet":
            normalize = lambda x: x * 255 - 117
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        img_size = 229 if model_name == "googlenet" else 224

        transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

    model_transforms['model_name'] = transformations
    return transformations


def get_dataloader(datadir: str, model_name: str, batch_size: Optional[int] = None):
    """
    Provides a dataloader for imagenet validation set.

    :param datadir: path to dataset, e.g. /path/to/imagenet/val
    """

    transform = get_transforms(model_name)

    dataset = ImageFolderWithPaths(datadir, transform)

    if batch_size is None:
        batch_size = 30 if model_name == "clip-resnet50" else 32
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8
    )

    return dataloader


def load_model(model_name):
    """Loads a model to device and puts it in eval mode."""

    if model_name == "googlenet":
        model = inceptionv1(pretrained=True)
    elif model_name == "clip-resnet50":
        model, _, _ = create_model_and_transforms('RN50', pretrained='openai')
    elif model_name.startswith("resnet50"):
        # we use our own resnet50 to get pre-relu activations
        model = resnet.resnet50()

        ckpt_url = ckpt_dict[model_name]
        checkpoint = torch.hub.load_state_dict_from_url(
            ckpt_url, map_location=DEVICE if DEVICE != "mps" else "cpu"
        )

        if model_name in ["resnet50-linf", "resnet50-l2"]:
            state_dict = {
                k[len("module.model.") :]: v
                for k, v in checkpoint["model"].items()
                if k[: len("module.model.")] == "module.model."
            }  # Consider only the model and not normalizers or attacker
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
    elif model_name == "wide_resnet50":
        model = resnet.wide_resnet50_2()
        ckpt_url = ckpt_dict[model_name]
        checkpoint = torch.hub.load_state_dict_from_url(
            ckpt_url, map_location=DEVICE if DEVICE != "mps" else "cpu"
        )
        model.load_state_dict(checkpoint)
    elif model_name == "densenet_201":
        model = torchvision.models.densenet201(
            weights=torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
        )
        # Replace ReLU with non-inplace ReLUs
        replace_relu(model)
    elif model_name == "in1k-vit_b32":
        model = timm.create_model("vit_base_patch32_224.augreg_in1k", pretrained=True)
    elif model_name == "clip-vit_b32":
        model, _, _ = create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
    elif model_name == "convnext_b":
        model = timm.create_model("hf-hub:timm/convnext_base.fb_in1k", pretrained=True)
        # No need to replace ReLU since this model uses non-inplace GeLU units.
    else:
        raise KeyError(f"Model {model_name} not known!")

    # in any case, push model to device and set to eval mode
    model.to(DEVICE).eval()

    return model


def aggregate_activations(activations, model_name, layer_name):
    """
    Prepares the activations obtained at the layer for storage,
    to get tensor of shape batchsize x num_units.

    :param activations: the activations tensor
    :param model_name: the name of the model
    :param layer_name: the name of the layer
    :returns: tensor of activations of shape batchsize x num_units
    """

    # calculate the reduced mean across each channel
    if test_permute(model_name, layer_name):
        # convnext 1x1 convolutions implemented as linear layers 
        activations = torch.permute(activations, (0, 3, 1, 2))

    if model_name == "clip-vit_b32":
        # for clip-transformer, acts has shape seq_length x batch x embedding_dim
        activations = torch.mean(activations, dim=(0))
    elif model_name == "in1k-vit_b32":
        # for timm-transformer, acts has shape batch x seq_length x embedding_dim
        activations = torch.mean(activations, dim=(1))
    elif len(activations.shape) > 2:
        # activations has shape batchsize x out_channels
        activations = torch.mean(activations, dim=(-1, -2))

    return activations


def get_activations(model, model_name, layer, channel, images):
    """
    Gets the activations that a list of images (224x224 feature visualisations) achieves at a unit.

    :param model: the model
    :param layer: the layer of the target unit
    :param channel: the channel of the target unit
    :param images: the list of images
    :returns: a list of activations (same order as images)
    """

    # Lucent applies InceptionTransform to googlenet.
    # Apart from that, irrespective of model-specific transforms,
    # FVs were generated using torchvision standard transforms

    transform_list = [
        transforms.ToTensor()
    ]
    if model_name == "googlenet":
        normalize = lambda x: x * 255 - 117
        transform_list.append(normalize)
    elif model_name in ("clip-resnet50", "in21k_in1k-vit_b32", "clip-vit_b32"):
        transform_list.append(
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        )
    else:
        # For RN50, WRN, densenet and convnext.
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(
        transform_list
    )

    # apply transforms to images and push them to device
    imgs = [transform(img).to(DEVICE).unsqueeze(0) for img in images]
    imgs = torch.cat(imgs, dim=0)

    # feed images through model, obtain activations
    with torch.no_grad() and ModelHook(model, layer_names=[layer]) as hook:
        _logits = model(imgs)
        acts = hook(layer)

    acts = aggregate_activations(acts, model_name, layer)

    # select channel
    acts = acts[:, channel].cpu()

    # return the activations for the images
    return [acts[i].item() for i in range(len(images))]


def read_pickled_activations_file(filepath, units):
    """
    Reads and parses a .pkl file of activations and returns it as a pandas DataFrame,
    with columns 0, ..., n_units, path.

    :param filepath: full path to .pkl file
    :param units: list of ints, the units to be loaded into the df

    :returns: pandas DataFrame
    """

    with open(filepath, "rb") as fhandle:
        # res_dict maps 'activations' to np array and 'paths' to list of paths
        res_dict = pickle.load(fhandle)

        # select only those units that are of interest
        activations = res_dict["activations"][:, units]

        # create dataframe from relevant units
        dataframe = pd.DataFrame(activations, columns=[str(u) for u in units])

        # add the column of paths
        dataframe["path"] = res_dict["paths"]

    return dataframe


def get_min_max_exemplar_activations(model_name, unit_name, use_csv=False):
    """
    Returns the activations achieved by minimally and maximally activating image.

    :param model_name: str, the name of the model
    :param unit: str, the unit for which to find exemplars
    :param use_csv: bool, whether the old CSV files should be used

    :returns: min, max activation
    """

    layer, unit = split_unit(unit_name)

    # construct path to csv-files
    activations_dir = os.path.join(STORAGE_DIR, "activations", model_name)
    assert os.path.exists(activations_dir), "Could not find directory with activations!"

    if use_csv:
        # get path to CSV of this layer and get df
        csv_path = os.path.join(activations_dir, layer + ".csv")
        assert os.path.exists(csv_path), f"Could not find path to csv: {csv_path}"
        unit_df = pd.read_csv(csv_path, usecols=[str(unit)])
    else:
        pkl_path = os.path.join(activations_dir, layer + ".pkl")
        assert os.path.exists(
            pkl_path
        ), f"Could not find path to pickle-file: {pkl_path}"
        unit_df = read_pickled_activations_file(pkl_path, [unit])

    min_act = unit_df.min().get(str(unit))
    max_act = unit_df.max().get(str(unit))

    return min_act, max_act


def get_label_translator():
    """
    Returns a function that transforms a batch of labels to the new label values for InceptionV1.

    old labels:
        https://raw.githubusercontent.com/rgeirhos/lucent/dev/lucent/modelzoo/misc/old_imagenet_labels.txt
    new labels:
        https://raw.githubusercontent.com/conan7882/GoogLeNet-Inception/master/data/imageNetLabel.txt
    """

    with open("old_imagenet_labels.txt", "r", encoding="utf-8") as fhandle:
        old_imagenet_labels_data = fhandle.read()
    with open("imagenet_labels.txt", "r", encoding="utf-8") as fhandle:
        new_imagenet_labels_data = fhandle.read()

    # maps a class index to wordnet-id in old convention
    old_imagenet_labels_map = {}
    for cid, l in enumerate(old_imagenet_labels_data.strip().split("\n")):
        wid = l.split(" ")[0].strip()
        old_imagenet_labels_map[wid] = cid

    # maps a class index to wordnet-id in new convention
    new_imagenet_labels_map = {}
    for cid, l in enumerate(new_imagenet_labels_data.strip().split("\n")):
        wid = l.split(" ")[0].strip()
        new_imagenet_labels_map[cid] = wid

    def remap_torch_to_tf_labels(y):
        """Map PyTorch-style ImageNet labels to old convention used by GoogLeNet/InceptionV1."""
        res = []
        for yi in y.cpu().numpy():
            zi = None
            wid = new_imagenet_labels_map[yi]
            if wid in old_imagenet_labels_map:
                zi = old_imagenet_labels_map[wid]
                res.append(zi)
            else:
                raise ValueError(f"Unknown class {yi}/{wid}.")

        return torch.tensor(res).to(y.device) + 1

    return remap_torch_to_tf_labels


def replace_module(
    model: nn.Module,
    check_fn: Callable[[nn.Module], bool],
    get_replacement_fn: Callable[[nn.Module], nn.Module],
):
    """Recursively replaces modules in model with new modules.

    Args:
        model: The model to replace modules in.
        check_fn: A function that takes a module and returns True if it should
            be replaced.
        get_replacement_fn: A function that takes a module and returns a new
            module to replace it with.
    """
    children = list(model.named_children())
    for name, value in children:
        if check_fn(value):
            new_value = get_replacement_fn(value)
            setattr(model, name, new_value)
        replace_module(value, check_fn, get_replacement_fn)


def replace_relu(model: nn.Module):
    """Replaces all ReLU modules in model with non-inplace ones.
    Args:
        model: The model to replace modules in.
    """
    replace_module(
        model, lambda x: isinstance(x, nn.ReLU), lambda _: nn.ReLU()
    )


def get_clip_zero_shot_classifier(model, model_name):
    """
    Turns a CLIP-model into an ImageNet1k zero shot classifier.

    :param model: the pytorch model
    :param model_name: the model name
    """

    class ClipArgs:
        """Clip expects some object that holds configuration."""

        def __init__(self, model_name):
            if model_name == "clip-resnet50":
                self.model = "RN50"
                self.pretrained = "openai"
            elif model_name == "clip-vit_b32":
                self.model = "ViT-B-32"
                self.pretrained = "laion2b_s34b_b79k"

            self.distributed = False
            self.horovod = False
            self.precision = "fp32"
            self.batch_size = 32
            self.device = DEVICE

    args = ClipArgs(model_name)

    return zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)


def get_clip_logits(model, classifier, images):
    """
    Obtains the class-level logits for clip-trained models.

    :param model: the pytorch model
    :param classifier: the zero-shot classifier
    :param images: the batch of images
    """

    image_features = model.encode_image(images)
    image_features = F.normalize(image_features, dim=-1)
    logits = 100. * image_features @ classifier

    return logits
