# Multi-Class Certainty Mapped Network (MCCM-Net) for High Precision Segmentation of High-Altitude Imagery

## Overview

Welcome to the official GitHub repository for the paper **"Multi-Class Certainty Mapped Network for High Precision Segmentation of High-Altitude Imagery"**. This repository contains all the PyTorch scripts and resources necessary to reproduce the experiments and results presented in the [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13263/1326309/Multiclass-certainty-mapped-network-for-high-precision-segmentation-of-high/10.1117/12.3041825.full).

## Results

MCCM-Net outperforms traditional models like U-Net and Attention-U-Net in terms of Intersection over Union (IoU) and other evaluation metrics. The table below summarizes the performance of various models on the RescueNet dataset:

| Model          | IoU  | Pixel Accuracy | Precision | Recall | F1    | Parameters | Dataset  |
|----------------|------|----------------|-----------|--------|-------|------------|----------|
| MCCM-Net       | 0.836| 0.902          | 0.9327    | 0.902  | 0.905 | 15.5M      | RescueNet|
| Attention U-Net| 0.789| 0.873          | 0.914     | 0.873  | 0.879 | 34.8M      | RescueNet|
| U-Net          | 0.783| 0.869          | 0.911     | 0.869  | 0.873 | 31M        | RescueNet|

The repository also includes Mass-Building and WHU reproduction of UANet.

In the image below you can see the uncertainty heat maps of the compared models.
![Alt text](https://github.com/tunahan-oguz/UncertaintyAwareMultiClassSegmentation/blob/main/img/combined_image.png?raw=true)
## Repository Contents

- `train.py`: Script to train the MCCM-Net model.
- `segtest.py`: Script to evaluate the trained models.
- `models/`: Directory containing the model definitions.
- `dataset.py/`: File containing dataset classes.
- `utils/`: Utility functions for data preprocessing and object initializations.

## Installation

To run the scripts in this repository, you'll need to have Python 3.10 or later installed. You can install the required packages using `pip`:

```sh
pip install -r requirements.txt && export PYTHONPATH=.
```

## Usage

### Training

To train the MCCM-Net model on RescueNet dataset, run:

```sh
python3 train_app/scripts/train.py train --data data/mccm-net.yml --project project_name 
```
This script will create the following directories in the workspace run/train/project_name/train/weights, and under this folder you will find the checkpoints of your training. Before starting the training, make sure to update the dataset root path in the YAML file.

### Evaluation

To evaluate a trained model, run:

```sh
python3 segtest.py --pth path_to_your_model_checkpoint --conf config_file_used_to_train_the_model --K num_classes
```

Optional command line arguments of segtest.py:
- --vis True if you want to see the generated masks and uncertainty maps by model, False (default) to just calculate the metrics.

## Acknowledgements

We would like to thank the authors of, UA-Net for their contributions to the field of deep learning for image segmentation, which greatly inspired this work.

<!-- ## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{your_paper,
  title={Multi-Class Certainty Mapped Network for High Precision Segmentation of High-Altitude Imagery},
  author={Tunahan Oğuz, Toygar Akgün},
  booktitle={SPIE Asia Pacific Remote Sensing},
  year={2024}
}
``` -->

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/tunahan-oguz/UncertaintyAwareMultiClassSegmentation/blob/main/LICENSE) file for details.

---

