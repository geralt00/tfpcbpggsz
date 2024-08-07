# TF-PCBPGGSZ

## Description
TF-PCBPGGSZ is a TensorFlow-based tool designed for enhancing amplitude and mass fits in the B->Dh, D->Kspipi modes. It utilizes TensorFlow's efficient computation handling to significantly improve the stability and accuracy of phase correction orders in particle physics research.


## Installation

### Prerequisites
- TensorFlow: Ensure TensorFlow is installed on your system.
- Cudatoolkit >= 11.5: Required for running the tool.

### Known Issues
- There may be compatibility issues with certain versions of TensorFlow. See the [issue report on GitHub](https://github.com/tensorflow/tensorflow/issues/63362#issuecomment-2016019354) for more details.

### Setup
Clone the repository and set up the required environment using Conda:

 
```bash
git clone https://github.com/shenghui/tf-pcgamma.git
cd tf-pcgamma
conda env create -f requirements.yml
```

### Install the package
```bash
pip install -e . --no-deps
```