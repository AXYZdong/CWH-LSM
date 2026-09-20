# CWH-LSM

Code for **Cross-scale Weight-Heterogeneous Liquid State Machine (CWH-LSM)**.

CWH-LSM builds a Liquid State Machine reservoir of **cross scales** (three liquid layers of different sizes) initialized with **heterogeneous weight distributions** (Gamma / Gaussian / Pareto / Cauchy, or hybrid). Input samples are encoded into multi-scale spike features, which are then classified by an **MLP** or **SOM** readout. Evaluated on MNIST, FSDD, and N-MNIST.

## Project Structure

```
CWH-LSM/
├── CWH_LSM.py                  # Build CWH-LSM reservoir, encode dataset into spike features
├── MLP.py                      # MLP readout trained on saved spike features
├── SD-SOM.py                   # SOM readout trained on saved spike features
├── Ridge_Regression.py         # Ridge Regression readout trained on saved spike features
├── Network/
│   └── liquid_state_network.py # LSM builder + weight initializers (Gamma/Gaussian/Pareto/Cauchy)
├── Dataset/
│   └── NMNIST.py               # N-MNIST loader (.mat/.h5, auto center-crop to 28x28) (Note: The MNIST and FSDD loaders are included as standard in the BindsNET framework.)
├── requirements.txt
└── README.md
```

## Installation

Requires Python ≥ 3.8 and PyTorch (CUDA optional but recommended).

```bash
conda create -n cwh-lsm python=3.10 -y
conda activate cwh-lsm

# PyTorch: pick the command matching your CUDA version from pytorch.org, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Usage

Two stages: (1) encode the dataset into spike features with the CWH-LSM reservoir, (2) train a readout on the saved features.

### Stage 1: CWH-LSM encoding

```bash
python CWH_LSM.py --mode MNIST --data_root ./data --cuda_name cuda:0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `MNIST` | Dataset: `MNIST` / `FSDD` / `N-MNIST` |
| `--data_root` | `./data` | Dataset root directory |
| `--output_neuron` | `900` | Total reservoir neurons; defines the 3-scale split (400→[50,100,250], 784→[100,300,384], 900→[100,300,500], 1600→[300,500,800], 2500→[500,800,1200]) |
| `--hybrid` | on | Hybrid mode: the three scales use Gamma / Gaussian / Pareto weights respectively |
| `--weight_mode` | `Gaussian` | Single weight distribution when `--no-hybrid`: `Gaussian` / `Gamma` / `Pareto` |
| `--time` | `250` | Simulation time (ms) |
| `--seed` | `0` | Random seed |
| `--cuda_name` | `cuda:1` | GPU device (use `cuda:0` on single-GPU machines) |
| `--no-train` | — | Encode the test split instead of train |
| `--no-plot` | — | Disable live spike plotting (faster) |

Datasets are expected under `--data_root`: `MNIST/` (auto-downloaded), `FSDD/` (audio files), `NMNIST/` (`.mat` / `.h5` files, auto-preprocessed by `Dataset/NMNIST.py`).

Outputs are saved to `LSM_OUT/{MNIST|FSDD|NMNIST}/*_all.pt`, ready to be consumed by the readout scripts.

### Stage 2a: MLP readout

```bash
python MLP.py --cuda_name cuda:0
```

Scans `LSM_OUT/{MNIST,FSDD,NMNIST}/*_all.pt` and trains an MLP on each. Keep `--n_neurons` consistent with stage 1's `--output_neuron`. Results (accuracy / precision / recall / F1) are summarized in `Train_MLP_Out/{dataset}_mlp_summary_{seed}.csv`; confusion matrices and training curves are saved with `--confusion_matrix` / `--plot`.

### Stage 2b: SOM readout

```bash
python "SD-SOM.py" --dataset_name MNIST --cuda_name cuda:0
```

Trains a SOM (`IncreasingInhibitionNetwork`) on each `*_all.pt` under `LSM_OUT/MNIST/`, evaluated with all-activity and proportion-weighting voting. Outputs go to `Train_Out/`, `acc/`, and `model_weights/`.

## Quick Start (MNIST example)

```bash
# Stage 1: encode
python CWH_LSM.py --mode MNIST --data_root ./data --cuda_name cuda:0 --no-plot

# Stage 2: readout
python MLP.py --cuda_name cuda:0 --plot --confusion_matrix
# or
python "SD-SOM.py" --dataset_name MNIST --cuda_name cuda:0 --no-plot
```
