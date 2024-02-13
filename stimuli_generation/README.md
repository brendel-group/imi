# Stimuli Generation


### Step 1
`sample_units.py` samples a number of units from the network by first sampling a layer uniformly and then sampling a channel from that layer. Those units are then written into a user-specified json-file, so that they can be used for the generation of natural and optimized stimuli. 

For example: `python sample_units.py --model_name resnet50-l2 --n_units 50 --filename units.json`

### Step 2
`collect_activations.py` walks over the entire ImageNet validation set and records the activations achieved by every image, for all layers of interest of the specified model. For a ResNet, the layers of interest are conv, batchnorm and shortcut layers. The activations are stored as (large) pickle files, where one file contains the activations for one layer.
See docstring for more options.

For example: `python collect_activations.py --model_name resnet50-l2 --units_file units.json`

After that, `extract_exemplars.py` extracts the 2\*99 top-activating validation set images for the specified units-file from the pickle-files. Those images, which will be the reference images later, are combined with 2\*11 query images sampled from the desired percentile of activations (see below). These two lists are sorted by activation, from least to most activating (reverse for minimally activating images). Then, this list is divided into 10 buckets (bucket 0 contains the 11 least activating images, bucket 1 contains the next 11 images etc). The contents of those buckets are shuffled randomly, and 11 batches are created, where each image in a batch is sampled from a distinct bucket (i.e. there are as many batches as there are images per bucket.) This yields images max_0 to max_9 for each batch, i.e. 10 images per batch. The final images in a batch range from min_9 to max_9, where min_9 is the image that achieves the strongest negative activation and max_9 is the one with the strongest positive activation.
These batches are stored in `./stimuli/{model_name}/{layer_name}/channel_{unit}/natural_images/batch_{0-9}`.
To allow for flexible selection of images, the percentile is defined via its start- and end-indices in an (ascendingly) sorted list of activations (of length 50,000). See the example below.

For example: `python extract_exemplars.py --model_name resnet50-l2 --units_file units.json --start_min 5000 --stop_min 5011 --start_max 44989 --stop_max 45000`
Note how we only specify these index-ranges for the 11 query images, not for the reference images.

### Step 3

`get_diverse_optimized_stimuli.py` generates optimized stimuli for the units. For each unit, 10 feature visualizations are calculated. Stimuli are saved to `./stimuli/{model_name}/{layer_name}/channel_{unit}/optimized_images/`.

For example: `python get_optimized_stimuli.py --model_name resnet50-l2 --units_file units.json`

### Step 4

(Moving over to tools/data-generation/)
`create_task_structure_json.py` creates a json-file describing the setup for each task. For task i, we only use batches `(i % num_batches) + 1` (i.e. only batches 1-10 are used, 0 is reserved for catch trials). Units are sampled in a somewhat involved manner to make sure that every unit appears in equally many trials that are spread over distinct participants, ensuring that there is no correlation between units and the participants that see them. Order of units within trials is random, should not systematically show some units earlier / later across tasks.

For example: `python create_task_structure_json.py -nt=20 -nb 10 -nc=3 -nh=50 -s=../../../stimuli-generation/stimuli/resnet50-l2 -o l2_robust.json -m=natural --seed=1`

### Step 5

`create_task_structure_from_json.py` writes the actual task-folders, containing the stimuli. For normal trials with natural stimuli, min_9 / max_9 are used as queries (these come from the last bucket in step 2, so they are among the most strongly activating images) and the other 9 images are used as references. For catch trials with natural stimuli, min_8 / max_8 are used as queries but still appear in the references, rendering the task trivial. For normal trials with optimized stimuli, the natural min_9 / max_9 are used as queries, while the optimized 9 images are used as references. For catch trials with optimized stimuli, the min_8 / max_8 optimized images are shown as queries while still appearing in the references.

For example: `python create_task_structure_from_json.py -t exp_data -i l2_robust.json -nr 9`