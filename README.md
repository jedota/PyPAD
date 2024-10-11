# PyPAD

This repository make availble the PyPAD toolkit for evaluated a Preseetation Attack Detection for ID Cards based on the paper: "Improving Presentation Attack Detection for ID-Cards on Remote Verification System" Sebastian Gonzalez and Juan Tapia (Under revision)


## Abstract

This paper developed an end-to-end Presentation Attack Detection system for a remote biometric ID Cards system based on MobileNetV2 and an updated concatenate approach applied to Chilean ID Cards. Several composite scenarios based on cropped and splice areas and different capture sources are
used, including bona fide and presentation attack species such as printed, display, composite, plastic (PVC), and synthetic ID card
images. This proposal was developed using a database consisting of 190.000 real case images with the support of a third-party
company. Also, a new framework called PyPAD was developed to estimate multi-class metrics that are compliant with the ISO/IEC 30107-3 standard
and will be made available for research purposes. Our method was trained on the source and border convolutional
neural networks separately. Border can reach BPCER100 1.69%, and Source reached 2.36% BPCER100 on ID Cards attacks. A two-stage concatenate model attack using a border and source networks together can reach 0.92% of BPCER100.


## Features

The PyPAD toolkit has the following features:
- It is fully compliant with ISO 30107-3 and configurable to choose and estimate results based on different thresholds.
- It is able to calculate metrics for binary and multi-class PAD systems.
- It can plot DET curves containing all the presentation attack species for comparison. The plot depicts the two operational points typically reported, BPCER10 and BPCER20, with values highlighted.
- An EER plot is automatically created, which can help us understand the relation between BPCER, APCER, and system thresholds.
- Kernel Distribution Estimation (KDE) plots are reported using a linear and a log scale to highlight details and thresholds.
- Configurable confusion matrices for multi-class problems and different thresholds.
- A summary report is automatically generated describing different operational points (BPCER, APCER). 


### Figure examples:

![Captura de pantalla 2024-07-03 a la(s) 12 24 09](https://github.com/jedota/PyPAD/assets/45126159/0ca69ade-9adf-43fa-ad12-00389dddb00e)


## How to use

1. Download (or build) the .whl PyPAD file (pip install -e PyPAD/)
2. Go to your project's virtual environment and install with pip: `pip install pypad-23.10.1-py3-none-any.whl`
    - If you are using conda, first install pip inside your conda environment: `conda install pip`, then install the .whl package
3. Use inside your project as follows: `from pypad.metrics.iso_30107_3 import apcer`, and call the function as `apcer(...)`


### How to build the .whl file:

1. Install "build" inside your environment using pip: `pip install build`
2. Then build the package: `python -m build`
3. The package will be available inside the `dist` folder


### Notes:

* You can find an example of some functions of this library at `example/pypad.ipynb`. Running this on yout system will require [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).
* The following packages are included as dependencies and will be installed if not already present on your environment: `scipy`, `sklearn`, `matplotlib`, `seaborn`, and `pandas`.


## Cited (in revision process)

Arxiv Version: https://arxiv.org/abs/2301.09542

## License

This toolkit is only for research purposes. For any commercial use, please contact [juan.tapia-farias@h-da.de](mailto:juan.tapia-farias@h-da.de).


## Future work

Add MAD metrics.
