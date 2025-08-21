# TF-PCBPGGSZ

## Documentation

Following the related docs, I will trying to keep updates.
[readthedocs](https://tfgammafitter.readthedocs.io/en/latest/)
[codimd](https://codimd.web.cern.ch/jfrfDd_VQoaVz6yYKGpMpw)

## Description
TF-PCBPGGSZ is a TensorFlow-based tool designed for enhancing amplitude and mass fits in the B->Dh, D->Kspipi modes. It utilizes TensorFlow's efficient computation handling to significantly improve the stability and accuracy of phase correction orders in particle physics research.


## Installation

### Setup
Clone the repository:

```bash
git clone https://github.com/shenghui/tf-pcgamma.git
cd tf-pcgamma

```

### Conda
Set up the required environment using Conda
```bash
conda env create -f requirements.yml
conda install --file requirements.txt
```

### Install the package
```bash
pip install -e . --no-deps
```

## Project Structure

tfpcbpggsz/
├── tfpcbpggsz/              # main source code
│   ├── amp/                 # amplitude analysis modules
│   ├── bes/                 # BES data handling
│   ├── generator/           # toy MC and event generation
│   ├── lhcb/                # LHCb-specific code
│   ├── external/            # external utilities
│   ├── core.py              # core fitting routines
│   ├── fit.py               # fit engine
│   ├── dalitz_pdfs.py       # Dalitz plot PDFs
│   ├── masspdfs.py          # invariant mass PDFs
│   ├── phasecorrection.py   # phase correction logic
│   ├── plotter.py           # plotting utilities
│   └── version.py
│
├── benchmark/               # benchmarking configs, scripts & results
├── canorman_B2DPI_misID/    # misID shape studies
├── canorman_Efficency/      # efficiency-related studies
├── canorman_InvMassFit/     # invariant mass fitting studies
│
├── docs/                    # documentation (Sphinx)
├── examples/                # example scripts & galleries
│
├── environment.yml          # conda environment definition
├── requirements.txt         # pip dependencies
├── pyproject.toml           # project metadata & build config
├── README.md
└── CONTRIBUTING.md