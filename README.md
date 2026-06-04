# QuBound

**QuBound** is a data-driven framework for predicting quantum output bounds on noisy quantum computers. The code in this repository supports experiments that learn time-dependent probability bounds from historical backend noise properties and quantum circuit execution data.

The current release contains the core scripts, selected example datasets, and a small saved model checkpoint for reproducing the included GHZ-style bound prediction workflow.

## Overview

Noisy quantum computers can show time-varying behavior because backend calibration properties such as gate errors, readout errors, and coherence times change over time. QuBound uses these backend properties together with circuit-level information to predict upper and lower bounds for quantum circuit output probabilities.

This repository includes:

- model training and evaluation code for bound prediction,
- data generation utilities based on circuit structure and backend properties,
- selected example datasets for IBM quantum backends,
- a small pretrained model checkpoint for the GHZ4 example.

## Repository structure

```text
.
├── main.py                         # Train and evaluate the QuBound prediction model
├── data_generator.py               # Generate circuit/backend-property datasets
├── fetch.py                        # Fetch historical IBM backend properties
├── best_model_state_GHZ4.pth       # Saved model checkpoint for the GHZ4 example
├── requirements.txt                # Python package requirements
├── datasets/
│   ├── ibmq_kolkata/
│   │   └── 10000shots/
│   │       ├── GHZ3_ibmq_kolkata.csv
│   │       ├── GHZ4_ibmq_kolkata.csv
│   │       ├── RB3_ibmq_kolkata.csv
│   │       └── VQE4_ibmq_kolkata.csv
│   └── ibmq_mumbai/
│       └── GHZ4_ibmq_mumbai_test.csv
└── properties/
    └── README.md                   # Instructions for backend property JSON files
```

## Requirements

Recommended environment:

- Python 3.9 or 3.10
- PyTorch
- Qiskit / Qiskit Aer
- Qiskit IBM Provider
- NumPy, Pandas, SciPy, Statsmodels, Scikit-learn, Matplotlib
- TorchMetrics
- SupermarQ

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The current scripts use some legacy Qiskit APIs, including `Aer`, `execute`, and fake backend classes such as `FakeMumbai` and `FakeKolkata`. For this reason, the provided `requirements.txt` uses a Qiskit 0.45-style environment. If you prefer to use Qiskit 1.x or newer, the Qiskit imports in the scripts may need to be updated.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-github-username>/QuBound.git
cd QuBound

python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start

To train and evaluate the default GHZ4 example:

```bash
python main.py
```

By default, `main.py` loads:

```text
datasets/ibmq_kolkata/10000shots/GHZ4_ibmq_kolkata.csv
```

The script trains the prediction model, saves the best model state to:

```text
best_model_state_GHZ4.pth
```

and evaluates the predicted probability bounds on the validation/test data.

## Dataset generation

To generate new datasets from circuit and backend-property information:

```bash
python data_generator.py
```

The data generation script expects historical backend property files under the `properties/` directory. Example expected filenames are:

```text
properties/ibmq_mumbai_properties.json
properties/ibmq_kolkata_properties.json
```

These files are not included in this repository because they are large. See `properties/README.md` for details.

## Fetching backend properties

The `fetch.py` script provides an example for fetching historical IBM backend properties through `qiskit-ibm-provider`.

Before running it, make sure your IBM Quantum account is configured locally. For example, your environment should already be able to initialize:

```python
from qiskit_ibm_provider import IBMProvider
provider = IBMProvider()
```

Then run:

```bash
python fetch.py
```

You may need to modify the backend name, date range, and output filename inside `fetch.py`.

## Notes

- This repository is a cleaned code release for sharing the QuBound implementation on GitHub.
- Large IBM backend property JSON files are intentionally excluded from version control. They should be stored separately, for example through GitHub Releases, Google Drive, Zenodo, or another dataset hosting service.
- IBM Quantum credentials, tokens, and local configuration files are not included and should never be committed to the repository.
- Some experiment settings are currently hard-coded in the scripts, such as the dataset path, number of qubits, selected circuit type, and training hyperparameters. Modify the constants in `main.py` and `data_generator.py` for new experiments.
- The included datasets and checkpoint are intended as lightweight examples. Full experimental reproduction may require the original backend property traces and additional generated datasets.

## Citation

If you use this repository in academic work, please cite the corresponding QuBound paper or add the citation information here once available.
