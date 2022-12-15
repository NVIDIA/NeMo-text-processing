**NeMo Text Processing**
==========================

**This repository is under development, please refer to https://github.com/NVIDIA/NeMo/tree/main/nemo_text_processing for full functionality.**

Introduction
------------

`NeMo Text Processing` is a Python package for text normalization and inverse text normalization.

This repository is under development, please refer to https://github.com/NVIDIA/NeMo/tree/main/nemo_text_processing for full functionallity.
See [documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/text_normalization/wfst/wfst_text_normalization.html) for details.


Documentation
-------------

[Text Processing (text normalization and inverse text normalization)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/text_normalization/intro.html).

Tutorials
---------
A great way to start with NeMo is by checking [one of our tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html).

Getting help with NeMo
----------------------
FAQ can be found on NeMo's [Discussions board](https://github.com/NVIDIA/NeMo/discussions). You are welcome to ask questions or start discussions there.


Installation
------------

### Conda virtual environment

We recommend setting up a fresh Conda environment to install NeMo.

```bash
conda create --name nemo_tn python==3.8
conda activate nemo_tn
```

(Optional) To use [hybrid text normalization](nemo_text_processing/hybrid/README.md) install PyTorch using their [configurator](https://pytorch.org/get-started/locally/). 

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
**_NOTE:_** The command used to install PyTorch may depend on your system.


###  Pip

Use this installation mode if you want the latest released version.
```
pip install nemo_text_processing
```

###  Pip from source

Use this installation mode if you want the a version from particular GitHub branch (e.g main).
```
pip install Cython
python -m pip install git+https://github.com/NVIDIA/NeMo-text-processing.git@{BRANCH}#egg=nemo_text_processing
```


### From source

Use this installation mode if you are contributing to NeMo.
```
    git clone https://github.com/NVIDIA/NeMo-text-processing
    cd NeMo-text-processing
    ./reinstall.sh
```

**_NOTE:_** If you only want the toolkit without additional conda-based dependencies, you may replace ``reinstall.sh`` with ``pip install -e .`` when your PWD is the root of the NeMo repository.
