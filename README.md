# TF-PCGAMMA

## Description
TF-PCGAMMA is a TensorFlow-based package developed to accelerate amplitude and mass fits for the B->Dh, D->Kspipi modes. This tool leverages TensorFlow's capabilities to handle large computations efficiently, aiming to improve the stability and accuracy of phase correction orders in particle physics research.

## Installation

### Prerequisites
Ensure you have TensorFlow installed on your system as it is the primary dependency for this package.

Cudatoolkit >= 11.5 
 
Please check if its work when tensorflow is installed, since some known issue happen see [[Issue report](https://github.com/tensorflow/tensorflow/issues/63362#issuecomment-2016019354)]

### Setup
To install TF-PCGAMMA, clone the repository and install the required dependencies listed in `requirements.yml`:

```bash
git clone https://github.com/shenghui/tf-pcgamma.git
cd tf-pcgamma
conda env create -f requirements.yml
