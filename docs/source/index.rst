Medical Meta Learner
====================

.. image:: _static/mml_logo.png

|Python Badge| |Pipeline Badge| |PyPI Badge| |Docs Badge| |Style Badge| |License Badge|

``mml`` is a research oriented Python package which aims to provide an easy and scalable
way of deep learning on multiple image tasks (see `Meta-Learning <https://arxiv.org/abs/2004.05439>`_).

It features:

  * a clear methodology to store, load, refer, modify and combine RGB image datasets across task types (classification, segmentation, ...)
  * a highly configurable CLI for the full deep learning pipeline
  * a dedicated file management system, capable of continuing aborted experiments, reuse previous results and parallelize runs
  * an api for interactive pre- and post-experiment exploration
  * smooth integration of latest deep learning libraries (`lightning <https://github.com/Lightning-AI/lightning>`_, `hydra <https://github.com/facebookresearch/hydra>`_, `optuna <https://github.com/optuna/optuna>`_, ...)
  * easy expandability via using plugins or directly hooking into runtime objects via scripts or notebooks
  * good documentation, broad testing and ambitious goals


.. note::
    MML is still considered in Beta stage, which means any feedback is highly appreciated!


Quickstart
----------
Setup ``mml`` as described in :doc:`install` and write a short script to load your data into ``mml`` as follows:

.. code-block:: python

    from mml.api import (DSetCreator, License, Keyword, TaskCreator, TaskType, register_dsetcreator,
                         register_taskcreator, get_iterator_and_mapping_from_image_dataset)
    from mml.cli import main
    # this example shows how to quickly include an existing pytorch image classification dataset
    from my_code.data import MyExistingPyTorchDataSet

    dset_name = 'my_dataset'
    task_name = 'my_task'

    @register_dsetcreator(dset_name=dset_name)
    def create_dset():
        dset_creator = DSetCreator(dset_name=dset_name)
        # DSetCreator has various help functions to create datasets (e.g. from kaggle, pytorch datasets, ...)
        train_dset = MyExistingPyTorchDataSet(root=dset_creator.download_path, download=True, train=True)
        test_dset = MyExistingPyTorchDataSet(root=dset_creator.download_path, download=True, train=False)
        dset_path = dset_creator.extract_from_pytorch_datasets(datasets={'training': train_dset,
                                                                         'testing': test_dset},
                                                               task_type=TaskType.CLASSIFICATION,
                                                               class_names=train_dset.classes)
        return dset_path


    @register_taskcreator(task_name=task_name, dset_name=dset_name)
    def create_task(dset_path: Path):
        task = TaskCreator(dset_path=dset_path, name=task_name,
                           task_type=TaskType.CLASSIFICATION,
                           desc="(optional) My task description.",
                           ref="(optional) My bibtex entry.",
                           url='(optional) My data website.',
                           instr='(optional) Any instructions to access data.',
                           lic=License.UNKNOWN,  # the license of the task
                           release='(optional) Year of data release.',
                           keywords=[Keyword.NATURAL_OBJECTS])  # choose from a variety of keywords to describe data background
        # if classes are split by folders (which is the case if using extract_from_pytorch_datasets), one may simply
        train_iterator, idx_to_class = get_iterator_and_mapping_from_image_dataset(
            root=dset_path / 'training_data', classes=None)
        test_iterator, _ = get_iterator_and_mapping_from_image_dataset(
            root=dset_path / 'testing_data', classes=None)
        task.find_data(train_iterator=train_iterator, test_iterator=test_iterator, idx_to_class=idx_to_class)
        task.auto_complete()

    # start the MML cli from this script
    if __name__ == "__main__":
        main()

You can run your script with any ``mml`` CLI configurations and use the registered data along.
The following command installs the data, preprocesses it, trains a model and infers predictions on the test split.

.. code-block:: bash

    python script_name.py create task_list=[my_task]
    python script_name.py pp task_list=[my_task]
    python script_name.py train pivot.name=my_task mode.subroutines=[train,predict] mode.cv=false

See :doc:`usage` for more details on customizing the pipeline via CLI. Note that after the ``create`` call (where
the registered creators are needed) from now on you may omit the ``python script_name.py`` to start the ``mml``
pipeline and instead type ``mml train ...`` instead.

Similar libraries
-----------------

Here is a small comparison to python packages that are close to ``mml``:
  * `lightning-hydra-template <https://github.com/ashleve/lightning-hydra-template>`_ is a template for deep learning projects, similarly relying on hydra and pytorch lightning, offers much less functionality and configuration options as it is intended to be individually extended for each project, ``mml`` on the other hand tries to unify many tasks, datasets and models in one environment to ease cross project reusability
  * `GaNDLF <https://github.com/mlcommons/GaNDLF>`_ the Generally Nuanced Deep Learning Framework for segmentation, regression and classification has a similar scope to ``mml``, no code requried to train robust models and few code to customize the framework, to name some differences it relies on click instead of hydra, implements training routines itself instead of leveraging pytorch lightning and focuses less on reusability of past experiments
  * `MONAI <https://github.com/Project-MONAI/MONAI>`_ provides state-of-the-art, end-to-end training workflows for healthcare imaging; it implements a lot of metrics, network architectures and transforms specifically to the need of 3D medical image segmentation (but is not limited to this use case), preserving meta information on model training and applicability is also part of the concept, training routine is based on ignite (in contrast to pytorch lignting in ``mml``)
  * `OpenMMLab <https://github.com/open-mmlab>`_ provides an ecosystem of dozens of interoperable toolboxes for computer vision models (e.g. `mmdetection <https://github.com/open-mmlab/mmdetection>`_ for detection models, `mmpose <https://github.com/open-mmlab/mmpose>`_ for pose estimation or `mmpretrain <https://github.com/open-mmlab/mmpretrain>`_ for model pre-training), while expandability and interoperability is a key feature it has minimal dependencies - implementing most features within the ecosystem

Author and Contributors
-------------------------

Feel free to leave **bug reports** or **feature requests**:

Main author (>99%):

- `Patrick Godau <https://www.dkfz.de/en/imsy/team/people/Patrick_Scholz.html>`_

Other contributors:

- `Akriti Srivastava <https://de.linkedin.com/in/akriti-srivastava-76041a120>`_
- `Leon Mayer <https://www.dkfz.de/en/imsy/team/people/Leon_Mayer.html>`_
- `Dominik Michael <https://www.dkfz.de/en/imsy/team/people/Dominik_Michael.html>`_
- `Piotr Kalinowski <https://www.dkfz.de/en/imsy/team/people/Piotr_Kalinowski.html>`_
- `Amine Yamlahi <https://www.dkfz.de/en/imsy/team/people/Amine_ElYamlahi.html>`_

Licensing
---------

This library is licensed under the permissive `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_,
which is fully compatible with both **academic** and **commercial** applications. This project is/was supported by

  * the German Federal Ministry of Health under the reference number 2520DAT0P1 as part of the `pAItient <https://www.bundesgesundheitsministerium.de/ministerium/ressortforschung/handlungsfelder/forschungsschwerpunkte/digitale-innovation/modul-3-smarte-algorithmen-und-expertensysteme/paitient>`_ (Protected Artificial Intelligence Innovation Environment for Patient Oriented Digital Health Solutions for developing, testing and evidence based evaluation of clinical value) project,
  * `HELMHOLTZ IMAGING <https://helmholtz-imaging.de/>`_, a platform of the Helmholtz Information & Data Science Incubator and
  * the Helmholtz Association under the joint research school `“HIDSS4Health – Helmholtz Information and Data Science School for Health" <https://www.hidss4health.de/>`_

If you use this code in a research paper, **please cite**:

::

        @InProceedings{Godau2021TaskF,
            author="Godau, Patrick and Maier-Hein, Lena",
            editor="de Bruijne, Marleen and Cattin, Philippe C. and Cotin, St{\'e}phane and Padoy, Nicolas and Speidel, Stefanie and Zheng, Yefeng and Essert, Caroline",
            title="Task Fingerprinting for Meta Learning inBiomedical Image Analysis",
            booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021",
            year="2021",
            publisher="Springer International Publishing",
            pages="436--446"
        }


.. |Pipeline Badge| image:: https://github.com/IMSY-DKFZ/mml/actions/workflows/full-CI.yml/badge.svg
   :target: https://github.com/IMSY-DKFZ/mml
.. |PyPI Badge| image:: https://img.shields.io/pypi/v/mml-core
   :target: https://pypi.org/project/mml-core/
.. |Python Badge| image:: https://img.shields.io/badge/python-3.10-informational
   :target: https://www.python.org/doc/versions/
.. |Docs Badge| image:: https://readthedocs.org/projects/mml/badge/?version=latest
   :target: https://mml.readthedocs.io/en/latest/
.. |Style Badge| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
.. |License Badge| image:: https://img.shields.io/badge/license-MIT-blue
   :target: https://opensource.org/license/mit/

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents

   install
   usage
   hpo
   plugins
   modes
   guides
   extensions
   cli/overview


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API

   api/components
   api/overview
   api/plugins/overview


Indices
=======

    * :ref:`genindex`
    * :ref:`modindex`
