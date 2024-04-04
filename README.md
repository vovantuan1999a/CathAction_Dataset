# CathAction_Dataset

<!-- end badges -->

> [CathAction](https://epic-kitchens.github.io/) , a large-scale dataset for catheterization understanding. Our
CathAction dataset encompasses approximately annotated frames for catheterization action understanding and collision detection, and groundtruth masks for catheter and guidewire segmentation.

## Authors

**Contact:** 


## Citing
When using the dataset, kindly reference:

```
```

(Check publication [here]())

## Dataset Details

### Ground Truth
We provide ground truth for action segments and object bounding boxes.

* **Objects collision detection:** We annotate the tip of the catheter (or guidewire) with a bounding box.
* **Actions:**These classes belong to three groups: catheter (advance catheter and retract catheter), guidewire (advance guidewire and retract guidewire), and one action involving both the catheter and guidewire (rotate).
* **Segmentation*:** We manually label the catheter and the guidewire class separately in our dataset.

### Dataset Splits
The dataset is comprised of three splits with the corresponding ground truth:

* Training set - Full ground truth.
* Test set - Start/end times only.

Initially we are only releasing the full ground truth for the training set in order to run action, object and segmentation.


### Important Files

* `README.md (this file)`
* `README.html`
* `README.pdf`
* [`license.txt`](#license)
* [`CathAction_train_action_labels.csv`](CathAction_train_action_labels.csv) ([Info](#CathAction_train_action_labels)) ([Pickle](CathAction_train_action_labels.pkl))
* [`CathAction_val_action_labels.csv`](CathAction_val_action_labels.csv) ([Info](#CathAction_val_action_labels)) ([Pickle](CathAction_val_action_labels.pkl))



### Additional Files

We also provide the RGB-features, Flow-feature in the: ...

## Files Structure

### CathAction_train_action_labels.csv
CSV file containing 14 columns:

| Column Name         | Type                         | Example          | Description                                                                                                           |
| ------------------- | ---------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| `uid`               | int                          | `6374`           | Unique ID of the segment.                                                                                             |
| `video_id`          | string                       | `P03_01`         | Video the segment is in.                                                                                              |
| `narration`         | string                       | `close fridge`   | English description of the action provided by the participant.                                                        |
| `start_timestamp`   | string                       | `00:23:43.847`   | Start time in `HH:mm:ss.SSS` of the action.                                                                           |
| `stop_timestamp`    | string                       | `00:23:47.212`   | End time in `HH:mm:ss.SSS` of the action.                                                                             |
| `start_frame`       | int                          | `85430`          | Start frame of the action (WARNING only for frames extracted as detailed in [Video Information](#video-information)). |
| `stop_frame`        | int                          | `85643`          | End frame of the action (WARNING only for frames  extracted as detailed in [Video Information](#video-information)).  |
| `participant_id`    | string                       | `P03`            | ID of the participant.                                                                                                |
| `verb`              | string                       | `close`          | Parsed verb from the narration.                                                                                       |
| `noun`              | string                       | `fridge`         | First parsed noun from the narration.                                                                                 |
| `verb_class`        | int                          | `3`              | Numeric ID of the parsed verb's class.                                                                                |
| `noun_class`        | int                          | `10`             | Numeric ID of the parsed noun's class.                                                                                |
| `all_nouns`         | list of string (1 or more)   | `['fridge']`     | List of all parsed nouns from the narration.                                                                          |
| `all_noun_classes` | list of int    (1 or more)   | `[10]`           | List of numeric IDs corresponding to all of the parsed nouns' classes from the narration.                             |

Please note we have included a python pickle file for ease of use. This includes
a pandas dataframe with the same layout as above. This pickle file was created with pickle protocol 2 on pandas version 0.22.0.

### CathAction_val_action_labels.csv
CSV file containing 14 columns:

| Column Name         | Type                         | Example          | Description                                                                                                           |
| ------------------- | ---------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| `uid`               | int                          | `6374`           | Unique ID of the segment.                                                                                             |
| `video_id`          | string                       | `P03_01`         | Video the segment is in.                                                                                              |
| `narration`         | string                       | `close fridge`   | English description of the action provided by the participant.                                                        |
| `start_timestamp`   | string                       | `00:23:43.847`   | Start time in `HH:mm:ss.SSS` of the action.                                                                           |
| `stop_timestamp`    | string                       | `00:23:47.212`   | End time in `HH:mm:ss.SSS` of the action.                                                                             |
| `start_frame`       | int                          | `85430`          | Start frame of the action (WARNING only for frames extracted as detailed in [Video Information](#video-information)). |
| `stop_frame`        | int                          | `85643`          | End frame of the action (WARNING only for frames  extracted as detailed in [Video Information](#video-information)).  |
| `participant_id`    | string                       | `P03`            | ID of the participant.                                                                                                |
| `verb`              | string                       | `close`          | Parsed verb from the narration.                                                                                       |
| `noun`              | string                       | `fridge`         | First parsed noun from the narration.                                                                                 |
| `verb_class`        | int                          | `3`              | Numeric ID of the parsed verb's class.                                                                                |
| `noun_class`        | int                          | `10`             | Numeric ID of the parsed noun's class.                                                                                |
| `all_nouns`         | list of string (1 or more)   | `['fridge']`     | List of all parsed nouns from the narration.                                                                          |
| `all_noun_classes` | list of int    (1 or more)   | `[10]`           | List of numeric IDs corresponding to all of the parsed nouns' classes from the narration.                             |

Please note we have included a python pickle file for ease of use. This includes
a pandas dataframe with the same layout as above. This pickle file was created with pickle protocol 2 on pandas version 0.22.0.

## File Downloads

Due to the size of the dataset we provide scripts for downloading parts of the dataset:

* [videos]() (750GB)
* [frames]() (320GB)
  * [rgb-frames]() (220GB)
  * [flow-frames]() (100GB)
* [collision object annotation images]() (80GB)

*Note: These scripts will work for Linux and Mac. For Windows users a bash 
installation should work.*

These scripts replicate the folder structure of the dataset release, found 
[here]().

If you wish to download part of the dataset instructions can be found
[here]().


## Video Information
Videos are recorded in 1080p at 59.94 FPS on a GoPro Hero 5 with linear field of
view. There are a minority of videos which were shot at different resolutions,
field of views, or FPS due to participant error or camera. These videos
identified using `ffprobe` are:

* 1280x720: `P12_01`, `P12_02`, `P12_03`, `P12_04`.
* 2560x1440: `P12_05`, `P12_06` 
* 29.97 FPS: `P09_07`, `P09_08`, `P10_01`, `P10_04`, `P11_01`, `P18_02`,
    `P18_03`
* 48 FPS: `P17_01`, `P17_02`, `P17_03`, `P17_04`
* 90 FPS: `P18_09`

The GoPro Hero 5 was also set to drop the framerate in low light conditions to
preserve exposure leading to variable FPS in some videos.  If you wish to
extract frames we suggest you resample at 60 FPS to mitigate issues with
variable FPS, this can be achieved in a single step with FFmpeg: 

```
ffmpeg -i "P##_**.MP4" -vf "scale=-2:256" -q:v 4 -r 60 "P##_**/frame_%010d.jpg"
```

where `##` is the Participant ID and `**` is the video ID.

Optical flow was extracted using a fork of
[`gpu_flow`](https://github.com/feichtenhofer/gpu_flow) made 
[available on github](https://github.com/dl-container-registry/furnari-flow).
 We set the parameters: stride = 2, dilation = 3, bound = 25 and size = 256.

# CathAction PyTorch Dataset

A PyTorch Dataset for the **CathAction** datasets.

In particular, it handles **frames** and **features** (the latter provided by the RULSTM repo [[link](https://github.com/fpv-iplab/rulstm)]) for both the **Action Recognition** and the **Action Anticipation** tasks.


# Action Recognition Usage Example

```python
# Imports
from torchvision import transforms
from input_loaders import ActionRecognitionSampler, FramesLoader, FeaturesLoader, PipeLoaders
from utils import get_ek55_annotation

# Create clip samples and clip loader
sampler = ActionRecognitionSampler(sample_mode='center', num_frames_per_action=16)
loader = PipeLoaders([
    FramesLoader(sampler, 'path/to/frames', fps=5.0, transform_frame=transforms.ToTensor()),
    FeaturesLoader(sampler, 'path/to/features', fps=5.0, input_name='obj'),
])
csv = get_ek100_annotation(partition='train') # Load annotations (dataframe)
ds = EpicDataset(csv, partition='train', loader=loader, task='recognition') # Create the EK dataset

# Get sample
sample = next(iter(ds))

"""
sample['uid'] -> int
sample['frame'] -> tensor of shape [C, T, H, W]
sample['obj'] -> tensor of shape [T, D]
sample['noun_class'] -> int
sample['verb_class'] -> int
sample['action_class'] -> int
"""

```

# Action Anticipation Usage Example

```python
# Imports
from torchvision import transforms
from input_loaders import ActionAnticipationSampler, FramesLoader, FeaturesLoader, PipeLoaders
from utils import get_ek55_annotation

# Create clip samples and clip loader
sampler = ActionAnticipationSampler(t_buffer=3.5, t_ant=1.0, fps=5.0)
loader = PipeLoaders([
    FramesLoader(sampler, 'path/to/frames', fps=5.0, transform_frame=transforms.ToTensor()),
    FeaturesLoader(sampler, 'path/to/features', fps=5.0, input_name='obj'),
])
csv = get_ek55_annotation(partition='train', use_rulstm_splits=True) # Load annotations (dataframe)
ds = EpicDataset(csv, partition='train', loader=loader, task='recognition') # Create the EK dataset

# Get sample
sample = next(iter(ds))

"""
sample['uid'] -> int
sample['frame'] -> tensor of shape [C, T, H, W]
sample['obj'] -> tensor of shape [T, D]
sample['mask'] -> tensor of shape [T]
sample['noun_class'] -> int
sample['verb_class'] -> int
sample['action_class'] -> int
"""

```

# Install

For the installation, you have to clone this repo and download the annotations as follows:

```sh
# Clone the project
$ git clone https://github.com/guglielmocamporese/epic-kitchens-dataset-pytorch.git ek_datasets

# Go to the project folder
$ cd ek_datasets

# Download the annotations
$ ./setup_annotations.sh

# Optionally run the usage example
$ python example.py
```


## License
All files in this dataset are copyright by us and published under the 
... , found 

This means that you must give appropriate credit, provide a link to the license,
and indicate if changes were made. You may do so in any reasonable manner,
but not in any way that suggests the licensor endorses you or your use. You
may not use the material for commercial purposes.

## Disclaimer
CathAction were collected as a tool for research in computer vision, however, it is worth noting that the dataset may have unintended biases (including those of a societal, gender or racial nature).
