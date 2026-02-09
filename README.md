# Anonymous code release for double blind review

This repository provides the implementation used to reproduce the experiments reported in the submitted manuscript (double blind review).

## 1. Repository structure
```text
.
├── Bemerged99.csv            # raw dataset file (if publicly shareable)
├── Datata/                   # processed datasets for multi step forecasting
│   ├── datat1/               # t+1 (one step ahead)
│   ├── datat2/               # t+2 (two steps ahead)
│   └── datat3/               # t+3 (three steps ahead)
├── train.py                  # training script (supports --data / --adjdata)
├── model.py
├── engine.py
├── util.py
├── timexer.py
├── requirements.txt
└── garage/                   # checkpoints (created at runtime; do not commit)
```

## 2. Requirements
- Python 3.9 (recommended)
- OS: Linux is recommended (other OSes may work depending on environment)

## 3. Installation
Create a clean environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### PyTorch installation note
If you encounter issues installing `torch/torchvision/torchaudio`, please install PyTorch using the official selector for your OS/CUDA version, then re run:

```bash
pip install -r requirements.txt
```

## 4. Data
This repository contains:
- `Bemerged99.csv`: raw dataset file used to generate processed datasets (if included and publicly shareable).
- `Datata/`: processed datasets for multi step prediction.

### Processed datasets (multi step forecasting)
Processed datasets are organized as:
- `Datata/datat1/`: t+1 (predict the next time step)
- `Datata/datat2/`: t+2 (predict two steps ahead)
- `Datata/datat3/`: t+3 (predict three steps ahead)

For each horizon folder, the training script expects:
- a data folder (used in this repo): `Xph7`
- an adjacency matrix file: `adj_mx.pkl`

Example:
- `Datata/datat1/Xph7`
- `Datata/datat1/Xph7/adj_mx.pkl`

## 5. Training
Before training, create the checkpoint directory (to avoid save errors):

```bash
mkdir -p garage
```

### Train on t+1
```bash
python train.py --data Datata/datat1/Xph7 --adjdata Datata/datat1/Xph7/adj_mx.pkl
python train.py --data Datata/datat1/Xph7 --adjdata Datata/datat1/Xdo7/adj_mx.pkl
```

### Train on t+2
```bash
python train.py --data Datata/datat2/Xph7 --adjdata Datata/datat2/Xph7/adj_mx.pkl
python train.py --data Datata/datat2/Xph7 --adjdata Datata/datat2/Xdo7/adj_mx.pkl
```

### Train on t+3
```bash
python train.py --data Datata/datat3/Xph7 --adjdata Datata/datat3/Xph7/adj_mx.pkl
python train.py --data Datata/datat3/Xph7 --adjdata Datata/datat3/Xdo7/adj_mx.pkl
```

(Optional) specify device (example uses GPU 0):
```bash
python train.py --device 0 --data Datata/datat1/Xph7 --adjdata Datata/datat1/Xph7/adj_mx.pkl
```

Outputs:
- Model checkpoints are saved under `garage/` (e.g., `*.pth`)
- Training logs (Loss, MAPE, RMSE, etc.) are printed to stdout

## 6. Evaluation / Inference
If evaluation is integrated into `train.py`, metrics will be printed during training.
Otherwise, run the provided evaluation script (if available) with the saved checkpoint.

The code reports common forecasting metrics such as:
- Loss
- MAPE
- RMSE

## 7. Reproducibility notes
- Randomness may cause small differences across hardware and runs (especially on GPU).
- Keep versions consistent with `requirements.txt` for best reproducibility.


