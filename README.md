# NEST: A Neuromodulated Small-world Hypergraph Trajectory Prediction Model

This repository contains the experimental implementation of NEST, a trajectory prediction model that utilizes neuromodulated small-world hypergraphs.

## Environment Setup

1. Create a Python environment (Python 3.8+ recommended):
```bash
conda create -n nest python=3.8
conda activate nest
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Main dependencies include:
- numpy >= 1.21.0
- PyTorch >= 1.10.0
- nuScenes devkit >= 1.1.10
- tqdm >= 4.65.0
- tensorboard >= 2.10.0

## Data Preparation

1. Download the nuScenes dataset
2. Place the dataset files in the `data/nuscenes` directory
3. Run the preprocessing script:
```bash
python tools/preprocess_nuscenes.py
```

## Training

To train the model:
```bash
python tools/train.py
```

## Directory Structure

```
├── data/
│   └── nuscenes/          # nuScenes dataset
├── tools/
│   ├── train.py           # Training script
│   └── preprocess_nuscenes.py  # Data preprocessing
└── requirements.txt       # Project dependencies
```