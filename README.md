# QuBound

Code package for **QuBound**, a data-driven quantum bound prediction workflow for noisy quantum computers.

## Overview

QuBound studies quantum bound prediction under temporal backend noise. The current code includes:

- `main.py`: model training and evaluation for bound/prediction experiments.
- `data_generator.py`: dataset generation from quantum circuits and backend properties.
- `fetch.py`: utility script for fetching IBM backend properties.
- `datasets/`: selected example datasets.
- `properties/`: placeholder folder for backend property JSON files.

## Repository structure

```text
.
├── main.py
├── data_generator.py
├── fetch.py
├── requirements.txt
├── datasets/
│   ├── ibmq_kolkata/
│   └── ibmq_mumbai/
└── properties/
    └── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train/evaluate the model on the included GHZ4 Kolkata dataset:

```bash
python main.py
```

Generate a dataset:

```bash
python data_generator.py
```

Note: `data_generator.py` expects large backend property files such as:

```text
properties/ibmq_mumbai_properties.json
properties/ibmq_kolkata_properties.json
```

These large JSON files are not included in this GitHub-ready package. Store them separately or distribute them through GitHub Releases, Google Drive, Zenodo, or Git LFS.

## Notes

- The original development environment appears to use older Qiskit-style imports such as `from qiskit import Aer, execute` and fake backends such as `FakeMumbai` / `FakeKolkata`. If you run into Qiskit version issues, use an older compatible Qiskit environment or update the imports for your current Qiskit version.
- The `.idea/` folder from PyCharm/JetBrains was removed because it is local IDE configuration.
- Large backend property JSON files were removed from the repo-ready package to keep the GitHub repository lightweight.

## Suggested citation

If this repository accompanies a paper or preprint, add the citation here.

```bibtex
@misc{qubound,
  title  = {QuBound: Quantum Bound Prediction on Noisy Quantum Computers},
  author = {Your Name},
  year   = {2026},
  note   = {Code repository}
}
```
