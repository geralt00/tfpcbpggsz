# TF-PCGAMMA

## Description
TF-PCGAMMA is a TensorFlow-based tool designed for enhancing amplitude and mass fits in the B->Dh, D->Kspipi modes. It utilizes TensorFlow's efficient computation handling to significantly improve the stability and accuracy of phase correction orders in particle physics research.

### Key Components
- **analysis/tf_fit_autograd.py**: Main example fit script, recently optimized for better memory management.
- **analysis/read_result*.ipynb**: Jupyter notebooks for output analysis using iminuit.
- **func/**: Directory containing mass-related functions.
- **Core/**: Stores the fitted functions, Legendre functions, and a new bias generator (planned).
- **Example/**: Directory planned for clearer demonstrations in future updates.

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