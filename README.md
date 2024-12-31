<div align="center">

# Medical Meta Learner

![mml_logo.png](docs%2Fsource%2F_static%2Fmml_logo.png)

(Please note: badges are not yet finalized and are currently prepared for GitHub transition)

[![docs status](https://readthedocs.org/projects/mml/badge/?version=develop)](https://mml.readthedocs.io/en/develop/?badge=develop)
![build status](https://github.com/IMSY-DKFZ/mml/actions/workflows/<WORKFLOW_FILE>/badge.svg)
[![pypi Badge](https://img.shields.io/pypi/v/mml-core)](https://pypi.org/project/mml-core/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
![docker status](https://github.com/<OWNER>/<REPOSITORY>/actions/workflows/<WORKFLOW_FILE>/badge.svg) 
<br>
[![Python](https://img.shields.io/pypi/pyversions/mml-core.svg)](https://pypi.org/project/mml-core)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/) <br>
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
CodeCov
Pylint

</div>

<br>

## About

`mml` is a research-oriented Python package which aims to provide an easy and scalable
way of performing deep learning on multiple image tasks (see 
[Meta-Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9428530)).

It features:
  * a clear methodology to store, load, refer, modify and combine RGB image datasets across task types (classification, segmentation, ...)
  * a highly configurable CLI for the full deep learning pipeline
  * a dedicated file management system, capable of continuing aborted experiments, reuse previous results and parallelize runs
  * an api for interactive pre- and post-experiment exploration
  * smooth integration of latest deep learning libraries ([lightning](https://github.com/Lightning-AI/lightning), [hydra](https://github.com/facebookresearch/hydra), [optuna](https://github.com/optuna/optuna), ...)
  * easy expandability via plugins or directly hooking into runtime objects via scripts or notebooks
  * good documentation, broad testing and ambitious goals

Please read the [official documentation page](https://mml.readthedocs.io/en/latest/index.html) for more.
Main author: Patrick Godau, Deutsches Krebsforschungszentrum (DKFZ) Heidelberg

Division of Intelligent Medical Systems

Contact: patrick.godau@dkfz-heidelberg.de

## Setup


Create a virtual environment (e.g. using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)) as follows:

```commandline
conda create -n mml python=3.10
conda activate mml
```

Now create a gitlab access token via [this link](https://git.dkfz.de/-/user_settings/personal_access_tokens?name=mmlToken&scopes=read_api), 
everything is already pre-filled, simply create and copy the token. Afterward within your projects virtual environment 
you can call

```commandline
pip install --index-url https://mmlToken:<personal_access_token>@git.dkfz.de/api/v4/projects/89/packages/pypi/simple mml-core
```

and replace <personal_access_token> with your actual token. You may add extras appended to mml as described [below](#extras).

### plugins

Plugins extend `mml` functionality. See [here](https://mml.readthedocs.io/en/latest/api/plugins/overview.html) for a 
list of available plugins. Use the previous `pip` and replace `mml-core` with one of the plugins to install.

### local environment variables

`mml` relies on a `mml.env` file for relevant environment variables. There are multiple possibilities to locate this:

 - within your project folder (e.g. for separation of `mml` installations),
 - within your home folder or similar (e.g. for shared `mml` configs across installations)

You can use `mml-env-setup` from the command line at the location you want to place your `mml.env` file:

```commandline
mml-env-setup
```

Now you only need to pinpoint `mml` to your `mml.env` file. This can be done via an environment variable `MML_ENV_PATH` 
that needs to be present in the environment before starting `MML`. If you use conda this simplifies to 

```commandline
conda env config vars set MML_ENV_PATH=/path/to/your/mml.env
# if your file is located at the current working directory, you may instead use
# pwd | conda env config vars set MML_ENV_PATH=$(</dev/stdin)/mml.env
# either way this requires re-activation of environment
conda activate mml
# test if the path is set
echo $MML_ENV_PATH
```

You should see your path printed - if yes, continue providing the actual variables:

 - open `mml.env` in your preferred editor
 - set `MML_DATA_PATH` to the path you want to store downloaded or generated datasets later on
 - set `MML_RESULTS_PATH` to be the location you want to save your experiments in later on (plots, trained network parameters, caluclated distances, etc.).
 - set `MML_LOCAL_WORKERS` to be the number of usable (virtual) cpu cores
 - all other variables are optional for now

### Confirm installation
You can confirm that `mml` was installed successful via running `mml` in the terminal, 
which should result in a display of an MML logo.

## License

This library is licensed under the permissive [MIT license](https://en.wikipedia.org/wiki/MIT_License),
which is fully compatible with both **academic** and **commercial** applications.

Copyright German Cancer Research Center (DKFZ) and contributors. 
Please make sure that your usage of this code is in compliance with its [license](LICENSE.txt). 
This project is/was supported by 

- (i) the German Federal Ministry of Health under the reference number 2520DAT0P1 as part of the
[pAItient (Protected Artificial Intelligence Innovation Environment for Patient
Oriented Digital Health Solutions for developing, testing and evidence based
evaluation of clinical value)](https://www.bundesgesundheitsministerium.de/ministerium/ressortforschung/handlungsfelder/forschungsschwerpunkte/digitale-innovation/modul-3-smarte-algorithmen-und-expertensysteme/paitient) project, 
- (ii) [HELMHOLTZ IMAGING](https://helmholtz-imaging.de/), a platform of the Helmholtz Information & Data Science Incubator and 
- (iii) the Helmholtz Association under the joint research school [“HIDSS4Health – Helmholtz 
Information and Data Science School for Health"](https://www.hidss4health.de/)

If you use this code in a research paper, **please cite**:

```
@InProceedings{Godau2021TaskF,
    author="Godau, Patrick and Maier-Hein, Lena",
    editor="de Bruijne, Marleen and Cattin, Philippe C. and Cotin, St{\'e}phane and Padoy, Nicolas and Speidel, Stefanie and Zheng, Yefeng and Essert, Caroline",
    title="Task Fingerprinting for Meta Learning inBiomedical Image Analysis",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021",
    year="2021",
    publisher="Springer International Publishing",
    pages="436--446"
}
```
