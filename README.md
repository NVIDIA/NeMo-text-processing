**NeMo Text Processing**
==========================

**This repository is under development, please refer to https://github.com/NVIDIA/NeMo/tree/main/nemo_text_processing for full functionality. See [documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/text_normalization/wfst/wfst_text_normalization.html) for details.**

Introduction
------------

`nemo-text-processing` is a Python package for text normalization and inverse text normalization.

Documentation
-------------

[NeMo-text-processing (text normalization and inverse text normalization)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/text_normalization/intro.html).

Tutorials
-----------------

| Google Collab Notebook      | Description |
| ----------- | ----------- |
| [Text_(Inverse)_Normalization.ipynb](https://github.com/NVIDIA/NeMo-text-processing/blob/main/tutorials/Text_(Inverse)_Normalization.ipynb)     | Quick-start guide       |
| [WFST_Tutorial](https://github.com/NVIDIA/NeMo-text-processing/blob/main/tutorials/WFST_Tutorial.ipynb)   | In-depth tutorial on grammar customization        |


Getting help
--------------
If you have a question which is not answered in the [Github discussions](https://github.com/NVIDIA/NeMo-text-processing/discussions), encounter a bug or have a feature request, please create a [Github issue](https://github.com/NVIDIA/NeMo-text-processing/issues). We also welcome you to directly open a [pull request](https://github.com/NVIDIA/NeMo-text-processing/pulls) to fix a bug or add a feature.


Installation
------------

### Conda virtual environment

We recommend setting up a fresh Conda environment to install NeMo-text-processing.

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

Use this installation mode if you are contributing to NeMo-text-processing.

```
git clone https://github.com/NVIDIA/NeMo-text-processing
cd NeMo-text-processing
./reinstall.sh
```

**_NOTE:_** If you only want the toolkit without additional conda-based dependencies, you may replace ``reinstall.sh`` with ``pip install -e .`` with the NeMo-text-processing root directory as your current working director.


Citation
--------

```
@inproceedings{zhang21ja_interspeech,
  author={Yang Zhang and Evelina Bakhturina and Boris Ginsburg},
  title={{NeMo (Inverse) Text Normalization: From Development to Production}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={4857--4859}
}

@inproceedings{bakhturina22_interspeech,
  author={Evelina Bakhturina and Yang Zhang and Boris Ginsburg},
  title={{Shallow Fusion of Weighted Finite-State Transducer and Language Model for
Text Normalization}},
  year=2022,
  booktitle={Proc. Interspeech 2022}
}
```

License
-------
NeMo-text-processing is under [Apache 2.0 license](LICENSE).
