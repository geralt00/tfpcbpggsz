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
```

### Install the package
```bash
pip install -e .
```

## Examples

### Ex1 Generate the Quantum-correlated data with phase-correction applied

```python
python3 ex1_gen_toy.py --config config_toy.yml --fit_result results/ampgen/fit_result_order6.json --order 6
```

Then the toy will save as a root file in the specified data path.

### Ex2 Fit the QC toy data

```python
python3 ex2_fit_toy.py --config config_toy.yml --order 6 --plot --plot-all --plot-each

```


### Ex3 Fit the QC real data

```python
python3 ex3_fit_bes_data.py --config config.yml --order 6 --plot --plot-all --plot-each

```
