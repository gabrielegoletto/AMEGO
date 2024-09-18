# AMEGO: Active Memory from long EGOcentric videos

<font size='5'><a href="https://arxiv.org/abs/2409.10917">**AMEGO: Active Memory from long EGOcentric videos**</a></font>

Gabriele Goletto, Tushar Nagarajan, Giuseppe Averta, Dima Damen

<a href='https://gabrielegoletto.github.io/AMEGO/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2409.10917'><img src='https://img.shields.io/badge/Paper-Arxiv:2409.10917-red'></a>

<center>
<figure>
    <img src="assets/teaser.png" width="900px" />
    <figcaption>
    This project provides tools for extracting and processing location segments and hand-object interaction tracklets in egocentric videos, the AMEGO representation.
    </figcaption>
</figure>
</center>

## Getting Started

### Installation

**1. Clone the Repository and Set Up Environment**

Clone this repository and create a Conda environment:

```bash
git clone --recursive https://github.com/yourusername/AMEGO
cd AMEGO
conda env create -f amego.yml
conda activate amego
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
cd submodules/epic-kitchens-100-hand-object-bboxes
python setup.py install
```

Download Tracker Weights

Download the weights from [this link](https://drive.google.com/file/d/14vZmWxYSGJXZGxD5U1LthvvTR_eRzWCw/view) and save them in `model_checkpoints/`.

The expected data stucture for EPIC-KITCHENS videos is:
<root>
│
├── EPIC-KITCHENS/
│   ├── <p_id>/
│   │   ├── rgb_frames/
│   │   │   └── <video_id>/
│   │   │       ├── frame_0000000000.jpg
│   │   │       ├── frame_0000000001.jpg
│   │   │       └── ...
│   │   │
│   │   ├── flowformer/
│   │   │   └── <video_id>/
│   │   │       ├── flow_0000000000.pth
│   │   │       ├── flow_0000000001.pth
│   │   │       └── ...
│   │   │
│   │   └── hand-objects/
│   │       └── <video_id>.pkl
│   │
│   └── ...
│
└── ...


The expected data structure for new videos is:
<root>
│
├── <video_id>/
│   ├── rgb_frames/
│   │   ├── frame_0000000000.jpg
│   │   ├── frame_0000000001.jpg
│   │   └── ...
│   │
│   ├── flowformer/
│   │   ├── flow_0000000000.pth
│   │   ├── flow_0000000001.pth
│   │   └── ...
│   │
│   └── hand-objects/
│       └── <video_id>.pkl
│
└── ...

**2. (Optional) Extract optical flow**

**2a. Prepare Flowformer Model**

Download the Flowformer model trained on the Sintel dataset from [this link](https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3PoqRdokrgWsGIf3yBA_). Place the model files in `submodules/flowformer/models/`.

**2b. Extracting Flowformer Flow**

Run the following command to extract flow data:

```bash
python -m tools.generate_flowformer_flow --root <root> --v_id <video_id> --dset <epic|video> --models_root submodules/flowformer/models --model sintel
```

**3. (Optional) Extract HOI detections (already given for EPIC-KITCHENS videos)**

**3a. Download Hand-Object Model**

Download the model from [this link](https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE) and place it in `submodules/hand_object_detector/models/`.

**3b. Create and Activate Environment (an ad-hoc environment is required)**

```bash
conda create --name handobj python=3.8
conda activate handobj
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
cd submodules/hand_object_detector/
pip install -r requirements.txt
cd lib
python setup.py build develop
pip install protobuf==3.20.3
pip install imageio
```

**3c. Extracting Hand-Object Bounding Boxes**

Run Extraction Script

```bash
python -m tools.extract_bboxes --image_dir <root>/<video_id>/rgb_frames --cuda --mGPUs --checksession 1 --checkepoch 8 --checkpoint 132028 --bs 32
```

Format Bounding Boxes

```bash
mkdir -p <video_id>/hand-objects/
python -m submodules.epic-kitchens-100-hand-object-bboxes.src.scripts.convert_raw_to_releasable_detections <video_id>.pb2 <video_id>.pkl --frame-height <video_height> --frame-width <video_width>
```

If there are issues with the `detections_pb2` file, run:

```bash
protoc -I ./tools/detection_types/ --python_out=. ./tools/detection_types/detections.proto
```

### Running AMEGO extraction

AMEGO extraction can be customized by adjusting configuration parameters. You can modify the configuration either by directly changing the values in the [default.yaml](./configs/default.yaml) file or by passing arguments via the command line interface (CLI).

**1. Extract Interaction Tracklets**

```bash
python HOI_AMEGO.py
```

**2. Extract Location Segments**

```bash
python LS_AMEGO.py
```


#### Output Structure

##### Interaction Tracklets

The output for interaction tracklets is saved as a JSON file with the following fields:

- **`track_id`**: The unique identifier for each interaction track.
- **`obj_bbox`**: The bounding box of the object involved in the interaction. The bounding box is not normalized, and coordinates are in **xywh** format (x-coordinate, y-coordinate, width, height) relative to the frame dimensions.
- **`num_frame`**: The list of frames where the object is detected during the interaction.
- **`features`**: DINO features extracted for the interaction track, providing detailed object representations.
- **`cluster`**: The object instance assigned to the track, which helps group similar object interactions.
- **`last_frame`**: The final frame in which the interaction is considered active.
- **`side`**: For each frame, this field contains information on the side of the hand (left or right) interacting with the object.

##### Location Segments

The output for location segments is also saved as a JSON file, but with a simplified structure containing the following fields:

- **`cluster`**: The object instance assigned to the segment, representing a unique object or cluster of objects.
- **`features`**: DINO features extracted for the segment, providing visual representations of the object.
- **`num_frame`**: The list of frames where the object appears in the segment.

Both can be easily read using `pandas` (`pd.read_json(<filename>)`). 

### Querying AMEGO for AMB

To query AMEGO for AMB:

```bash
python -m querying.Q* --root <root> --AMEGO <AMEGO root>
```
Replace `*` with the query type number ranging from 1 to 8 depending on the specific query.

## Acknowledgements

This repository builds upon previous works, specifically [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official) for optical flow extraction, [HOI Detector](https://github.com/ddshan/hand_object_detector) for hand-object interactions detection, and [EgoSTARK](https://github.com/EGO4D/episodic-memory/tree/main/EgoTracks) for tracking.

## BibTeX

If you use AMEGO in your research or applications, please cite our paper:

```bibtex
@inproceedings{goletto2024amego,
    title={AMEGO: Active Memory from long EGOcentric videos},
    author={Goletto, Gabriele and Nagarajan, Tushar and Averta, Giuseppe and Damen, Dima},
    booktitle={European Conference on Computer Vision},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.