# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.1.0 (XX/XX/2025)

### Features
 - persistent model checkpoints now only rely on the MML format (instead of previously using the lightning format)
 - improved error handling with interrupted dataset creation, which gives clear instructions how to resolve
 - new config group `peft`, which offers to inject `huggingface/peft` adapters into models and reduce train parameters

### Bug fixes
 - fixed incorrect handling of hydra choices in CONTINUE mode
 - fixed .env loading for non default system 

## 1.0.4 (05/02/2025):
This patch transitions the testing strategy from GPU based to purely CPU tests. 

### Bug fixes
 - make dataloader pin_memory depend on lightning accelerator
 - cast metric datatype to avoid occasional errors

### Features
 - make tests fully cpu compatible
 - speed up fake task creation by avoiding infer stats
 - add optional kwarg to task creator autocomplete that determines the device

## 1.0.3 (03/25/2025):
Post release fixes.

### Bug fixes
 - create default project in `src.mml.interactive.loading.default_file_manager`
 - fix documentation for `mml-drive` and `mml-lsf` plugins
 - relax class_occurrences verification in `TaskCreator` (now compatible with `shrink_train` tag from `mml-tags`)
 - fix `tl` mode model loading with new default value for `torch.load` in torch 2.6 (also generically for `mml.core.models.torch_base.BaseModel.load_checkpoint`)

## 1.0.2 (01/10/2025):
Post release fixes. Remove outdated installation instructions.

## 1.0.1 (12/31/2024):
Post release fixes. Renaming of `mml-data` to `mml-tasks` and updating some urls.

## 1.0.0 (12/19/2024):
The public release candidate for `mml`.

### Features
 - `use_best_params` now also accepts non-sql based multirun results
 - extended documentation
 - the initialization of `mml.interactive` is now guarded with helpful error messages
 - open file limit errors are also guarded with helpful error messages (#126)
 - refactors the `template.env` file with better documentation (and renaming)

### Bug fixes
 - `use_best_params` can now deal with config groups
 - template plugin was refactored to avoid ??? in paths, ensuring Windows compatibility (#145)
 - minor fixes in README, CONTRIBUTING and docs

## 0.14.2 (11/15/2024):
This release publishes final preparations for the release of `mml`.

### Features
 - full CLI documentation and increased documentation coverage
 - load function for `mml.interactive.AllTasksInfos`
 - refactored the notification system and introduced `mml.core.scripts.notifier.BaseNotifier`
 - meta file polishing
 - ensure MIT license compatibility of plugins

### Bug fixes
 - `mml.core.scripts.utils.StrEnum` to be creatable from both key and value

### API changes
 - removed the `plotting` config group and added the `info` mode related settings to that file

## 0.14.1 (09/23/2024):
This release publishes work in progress of the `mml` documentation.

### Features
 - added `sphinxcontrib-autoyaml` dependency and started CLI documentation in config files  

## 0.14.0 (08/29/2024):
This release introduces postprocessing and also extends prediction behaviour to comply with the `mml-suggest` plugin. 
Other highlights are the extended reusability functionality, test time augmentation and the outsourcing of plugin 
changelogs.

### Features
 - NEW: postprocessing scheduler - including quantification, calibration and automated ensembling
 - NEW: test time augmentation in testing and predicting, activate e.g. via `tta=rotate`
 - NEW: multi-project reusables (`reuse.parameters=[SOME_PROJ,OTHER_PROJ]`), global reusables (`MMLFileManager.instance().global_reusables`) and selecting specific version numbers (`reuse.parameters=SOME_PROJ#3`)
 - extended predicting behaviour: now on validation, test and unlabelled split
 - paths in model storage became relative - easing transferability of results between systems
 - simplify the creating and loading of pipeline configurations
 - new artefact: each run with return value produces a `return_val.txt` file at the end of `mml`
 - disable progress bar in cluster mode
 - individual CHANGELOG.md per plugin 
 - add test and unlabelled data to fake task

### Bug fixes
 - fix model checkpointing - now behaving correctly as documented and reducing training time
 - fix incorrect detection of lightning tuning in lightningmodule
 - fix progress bar import - now correctly displaying experiments 
 - fix some tests, including a better capture of warnings 
 - fix tarfile vulnerability in data archive extraction
 
### API changes
 - `reuse.clean_up` was moved to `remove`, so `reuse.clean_up.parameters=true` becomes `remove.parameters=true`
 - `model_storage.py` moved from `mml.core.data_loading` to `mml.core.scripts`
 - `find_data` in task creation now relies on `DataSplit` instead of string
 - top level config `gpus` became `allow_gpu` (regulating non training usage of GPUs)

## 0.13.3 (08/23/2024):
This release fixes a critical bug in preprocessing mode.

### Bug fixes
 - in the previous release preprocessing incorrectly produced corrupted images due to an obsolete float conversion 

## 0.13.2 (08/07/2024):
This release has two minor features added to model training/evaluation as well as some bug fixes.

### Features
 - loss weights may be passed from CLI (`loss.class_weights=[...]`)
 - allow `mode.eval_on` to accept multiple tasks in `train` mode

### Bug fixes
 - `mml.interactive.planning.get_task_infos` deals properly with tagged tasks

### Plugins
 - mml-data fix incorrect import of `mml.interactive`
 - add mml-drive to index
 - mml-lsf worker setting is less greedy, exclusive process is dropped and connection is tested on startup

## 0.13.1 (06/27/2024):
This release focuses on improving LSF Plugin experience as well as some bug fixes.

### Bug fixes
 - remove some artefacts in the docs
 - remove inputtimeout dependency from upgrade scheduler
 - fix batch size tuning
 - fix randaugment augmentation
 - fix load_imagenet_aa augmentation

### Plugins
 - LSF runner has received info, resubmit and retrieve functionality

### Changed default values
 - default_trainer config has changed (e.g. min_epochs)
 - usage of persistent workers reintroduced, see lightning datamodule 

## 0.13.0 (06/21/2024):
This release comprises some restructurings, improved documentation and usability as well as intense preparations for 
a release (e.g. license headers, removal of non-compliant code parts). A major usage change is omitting the `mode=` 
part from CLI. Nice new features are regression task support, interactive job runners, support of torchvision 
transforms, dependency updates and many more!

### Features
 - the `mode=MODE` part of calling mml has been replaced by simply calling `mml MODE`
 - improved `mml --help` and plain `mml` cli calls
 - major restructurings of the package
   - decorators and exceptions have received individual modules `mml.core.scripts.decorators` (resp. `exceptions`)
   - `mml.api.interactive` moved to `mml.interactive`
   - introduced `AugmentationModules` in `mml.core.data_loading.augmentations`, now supporting `kornia`, `torchvision` and `albumentations`
 - support for `TaskType.REGRESSION`
 - more effective and consistent preprocessing of tagged tasks 
 - `ModelStorage` can now be updated (simply call `store()` without any arguments from a loaded `ModelStorage`)
 - `JobRunner` class for interactive `MMLJobDescription` usage
 - model head re-mapping, see `SingleFrameLightningModule.setup_redirection()`
 - added linear probing via `mml tl mode.freeze=True ...`
 - improved metrics tracker callback (now accessible through scheduler)
 - a nice WIP Bar indicating MML computations without any logged output
 - more modality loaders
 - switched from flake8 to ruff linter (plus formatter)
 - simplify `tox.ini`
 - added license header to all code files

### Bug fixes
 - mixup and cutmix callbacks
 - torch base checkpointing and backbone freezing
 - continue mode logging directory

### Plugins
 - `mml-similarity` has seen some major make-over to be ready for release
   - removed `dds` mode
   - is called through single mode "similarity" and new config group "distance"
   - fix some bugs that have been introduced through `mml-core` changes
   - plus fix return value by task groups
 - introduced `mml-drive` as IMSY internal feature for faster downloads
 - `m̀ml-lsf` has received a remote cluster job runner

## 0.12.0 (01/24/2024):
Mainly focuses on cleaning the MML backend. More generic task descriptions, additional Modality support as well as a 
generic model concept are at the core of the release. It is important to note, that existing databases from previous 
versions need to migrate (after upgrading `mml-core` run `mml mode=upgrade mode.version=0.11.1`) once. To undo this 
you may call (`mml mode=downgrade mode.version=0.11.1` BEFORE downgrading `mml-core` package).

### Features
 - a new dataclass `TaskDescription` to replace `current_meta` dictionary of `TaskCreator` (#98)
 - deprecate `TaskStruct.id` (#98)
 - improved support for future `Modality`'s, e.g. video processing (#20)
 - faster task tagging
 - preprocessing of test data to allow for e.g. nested tasks
 - a new mode `clean` to remove artefacts automatically and free disk space
 - new modes `upgrade`/`downgrade` to support migrating between versions of breaking compatibility (#98) 
 - extended `ModelStorage` to hold task and fold information as well as creation time
 - a new task tag `nested` to allow better validation strategies (#91)
 - updated documentation
 - cleaning of config artefacts
 - updated dependencies, including removal of extras `[optuna]` and `[jupyter]` adding them as core dependencies
 - a `mml --version` implementation (#97)
 - renamed task attribute `Tag` to `Keyword` to avoid confusion with task tagging 
 - stable support for joint cpu and gpu augmentations (#28)
 - finally a stable license (#46)
 - refactoring models, allowing much more generic future extensions plus supporting multi-task in `mml-core` (#47)
 - test time metric bootstrapping (#91)
 - extend `info` mode to show information on existing trained models
 - replaced `opt` scheduler by `train` scheduler, with cross-validation support (#91)
 - implemented `transfer` scheduler in `mml-core` for transfer learning setups as inherited from `train` (#24)
 - support for password-protected zip decryption during archive extraction

### Bug fixes
 - `MMLFileManager` now correctly determines latest file versions
 - use a temp log path if creating `MMLFileManager` on the fly

### Plugins
 - `mml-tags` and `mml-data` have improved CLI, test via `mml-tags` or `mml-data --help`
 - fixed some references in `mml-data`
 - `mml-similarity` has additional visualization options, a fixed distance_measure abbreviation determination
 - a new `mml-sql` plugin that supports SQL based hyperparameter optimization
 - removed `mml-multi-task`, `mml-self-supervised` and `mml-student` plugins
 - `mml-data` has received a new task `suncolondb-classification`

### Additional Contributors
 - Leon Mayer

## 0.11.1 (06/06/2023):

### Features
 - add path information to the notification system `mml.core.scripts.notifier.py`
 - `ModelStorage` now also holds references to the created predictions

### Bug fixes
 - docker handling in gitlab CI
 - made temporary task file creation more parallelism safe in statistics calculation of new tasks

### Plugins
 - `mml-data` update `idle_action_recognition` download path

## 0.11.0 (05/26/2023):

### Features
 - BETA: docker support (#17, #74)
 - BETA: kornia image augmentations support (#28, #75)
 - BETA: torch.compile support (#51)
 - mml.api.interactive can now be provided a `MML_ENV_PATH`
 - more efficient checkpointing strategy (#62)
 - gitlab CI efficiency optimization
 - scheduler post initialization hooks, allowing plugins to interactively adapt the configs

### Bug fixes
 - dependency version fixes
 - minor documentation errors
 - continue status command indexing

## Plugins
 - new plugin `mml-lsf` for DKFZ LSF cluster convenience features
 - `mml-similarity`
   - TSNE task visualization
   - additional distance task colorization criteria
 - `mml-tags`
   - fix plugin activation
   - `shrink_train` has become an 'incremental' subsetting strategy (controllable via seed option) 

### Additional Contributors
 - Marco Hübner
 - Dominik Michael
 - Piotr Kalinowski
 - Amine Yamlahi

## 0.10.2 (05/03/2023):

### Features
 - new `mml` logo
 - sped up task size calculations
 - additional test benchmarks
 - gitlab CI optimizations
 - cleaning of project files, e.g. previously cached notebooks

### Bug fixes
 - documentation errors

### Additional Contributors
 - Marco Hübner
 - Dominik Michael
 - Piotr Kalinowski
 - Amine Yamlahi

## 0.10.1 (04/18/2023):

### Features
 - add `mml-env-setup` console script

## 0.10.0 (04/17/2023):

`0.10` marks splitting `mml` into `mml-core` and various plugins. 

### Features
 - drastically improved docs, including runtime diagram, API documentation and a lot of guides for beginners
 - more integration tests
 - new `MML_ENV_PATH` environment variable to freely place your `.env` file 
 - README update
 - plugins are loaded before hydra config compilation to allow for modified search paths
 - file manager `clean_up` is now ready for parallel processing
 - automatically set `torch.set_float32_matmul_precision('high')` for better tensorcore usage
 - ansi art `mml` logo
 - a lot cleaner set of default config files
 - `mml.testing` and a `pytest` plugin to provide generic fixtures used in testing `mml` itself and plugins

### API changes
 - access task structs via `scheduler.get_struct` instead of `scheduler.task_factory.get_by_name`
 - lightning `accelerator` and `devices` support instead of previous `gpus`
 - `create`/`preprocess` and `optimization` have moved into `core`
 - `mml-lib-init` has been renamed to `mml-copy-conf`
 - removed `assets` and outsourced many others to plugins
 - removed `mode.id` requirement for configs
 - scheduler task dumping is now done automatically

### Bug fixes
 - some fixes related to the previous upgrades of `lightning`, `torchmetrics` and `torch` (logging, progress bar, ...)
 - error handling with unset notification env variables

## 0.9.0 (03/21/2023):

### Features
 - support python 3.9 and 3.10
 - support pytorch 2.0, lightning 2.0 and hydra 1.3
 - add `mml_data` and `mml_tags` plugins to outsource non-core components
 - refactored task creators to ensure more safety measures and better error messages, including an `auto_complete` method
 - added prediction subroutines to opt mode
 - added caching option for smaller datasets, use `sampling.enable_caching` and `sampling.cache_max_size` to configure
 - more tests, pylint analysis and badge

### API changes
 - dropped support for `AA` mode, may come back with `kornia` support
 - refactored task tagging, new separators are `+` (tags) and `?` (params)
 - minor renaming, e.g. 
   - `Scheduler` nas been renamed to `AbstractBaseScheduler`
   - `api.notebooks` bas been renamed to `api.interactive`
 - dropped support for dict-like task creators
 - lightning callbacks have been moved from `cfg.callbacks` to `cfg.cbs`

### Bug fixes
 - class balanced sampling for multi-task learning 

## 0.8.1 (01/31/2023):

### Features
 - added task tag `redistribute` to change relative data splits
 - added slack notifier, set env variable `MML_SLACK_WEBHOOK_URL` and activate `logging.slack` in the configs (#33)
 - added email notifiers, set a bunch of env variables and activate `logging.email` (#33)
 - increased test suite
 - increased type hints
 - added citation file `CITATION.cff`

### API changes
 - the scheduler's lock path became an attribute
 - renamed `catch_time` from `mml.core.scripts.utils`
 - split testing from returning task struct within `TaskStructFactory`'s `get_by_name`

### Bug fixes
 - fix incorrect generation of `breast_cancer_classification` task
 - fix incorrect computation / logging of validation and test metrics
 - some remaining `LeraningPhase` and `DataSplit` references

## 0.8.0 (12/16/2022):

### Features
 - added `mode=dim` for dimensionality estimation of tasks
 - a set of convenience features in `api.notebooks` for experiment planning
 - `deprecated` and `beta` decorators
 - visualization of predicted samples
 - logging of confusion matrix
 - increased test coverage
 - increased loading speed for `ModelStorage` by switching from `.yaml` to `.json`
 - better formatting of `warnings`
 - performance improvements with `persistent_workers`

### API changes
 - introduced `LearningPhase` and `DataSplit` for use in datamodule and models

### Bug fixes
 - progress bar version shows active step naming
 - loading of metric values when reusing models
 - fixed a bug that occurred if continue mode was started without a previous model checkpoint being instantiated
 - linked `batch_size` in `aa` mode correctly
 - some fixes regarding the new path assignment strategy of `MMLFileManager`

## 0.7.4 (09/21/2022):
Support `EMPTY_MASK_TOKEN` during `find_data()`

### Bug fixes
 - allow `EMPTY_MASK_TOKEN` in `TaskCreator`

## 0.7.3 (09/21/2022):
Speed improvements for task creation.

### Features
 - `EMPTY_MASK_TOKEN` for empty segmentation masks
 - faster fold checks

## 0.7.2 (09/21/2022):
Fixes incapability to handle non-archived "downloaded" folders.

### Features
 - support unpacking of folders during dset creation

## 0.7.1 (09/20/2022):
Fixes plugins usage of `mml`.

### Bug fixes
 - fixing incorrect iteration over `mml.plugins`
 - move plugin loading into hydra decorated main


## 0.7.0 (09/19/2022):
This release focuses on the future library usage of `mml`.

### Features
 - drastically improved documentation
 - `mml.api` for easier usage as a lib
 - `mixup` and `cutmix` callbacks
 - callbacks are now intended to be stacked in CLI e.g. `+callbacks=[cutmix,swa]`
 - file manager paths greatly supports extendability, see `add_path_assignment`
 - `mml.plugins` entry point for other packages

### Bug fixes
 - fixing task probabilities with different task sizes in `mode=multi`

## 0.6.1 (08/26/2022):
Minor immediate fixes regarding packaging via gitlab. 

### Bug fixes
 - gitlab CI deployment procedure
 - installation as a package fix for `private.env`

## 0.6.0 (08/26/2022):
This is a major step forward with MML. After moving to gitlab and setting up a complete CI infrastructure 
much more convenience and continuity is to be expected. Some changes may cause backward incompatibility issues
(more detailed: git root folder has been shifted, variables in the private env config file have changed and a newer 
hydra version is used)!
Thus it is recommended to install mml from scratch when switching version.

### Features
 - mml can be used as a library! See README.md for details
 - new `mode=multi` for multi task learning
 - new `mode=tl` for transfer learning
 - new `mode=ss` for self-supervised learning
 - new `mode=stud` for student learning
 - new task type `MULTILABEL_CLASSIFICATION` for multi label classification tasks (even with soft labels)
 - made setup fully declarative -> no `setup.py` anymore, but all `setup.cfg` and `pyproject.toml`
 - new tasks and datasets that are added to `mml/create_tasks/data/preparation/classification` and `mml/create_tasks/data/preparation/segmentation` do not need to be added to the respective `__init__.py` files any more
 - started with spinx documentation
 - added ensembling of task similarities: new mode `ens` with input like `mode.sources=[proj_one_fed,proj_two_mmd,proj_one_kld]`
 - added cholect45 tasks
 - task similarity modes (e.g. `fed`, `mmd`, ...) now have a return value to allow for hpo
 - errors are getting logged with backtrace, including the backtrace, warnings are logged as well
 - training time of models is logged automatically in model storage
 - new task tag --shrink_train, allows for keeping identical validation split in reduced tasks
 - extended `mode=info`

### API changes
 - task `diabetic_retinopathy_diagnosis` has been renamed as `aptos19_blindness_detection`
 - using the imagenet AA has become a pipeline component instead of a separate config file

### Bug fixes
 - added time information to active_step_naming to avoid tensorboard run overlay in `--multirun` calls
 - resolved bug in preprocessing mode, that skipped preprocessing if no pivot was given
 - some updated task creations, due to changed urls
 - fixed deprecated stuff from pytorch lightning, albumentations, hydra, ...
 - fixed race condition issued when multiple runs in parallel might request saving paths from file manager

### Known Issues
 - performance of segmentation models poorly (opt mode, voc12 task, dice loss)
 - cross system support for model storage paths


## 0.5.0 (03/22/2022):
 Mainly improved setup time, updated torch and torchvision version as well as a bunch of new tasks.

### Features
 - `mode=info` was extended by subroutine `sample_grid`, producing a grid view of one sample per task
 - Faster MML setup time :) Introducing `load_meta_header` in `MMLFileManager` only partially parses .json files
 - dropped torchmetrics version restriction, so feel free to use more recent versions
 - started with some example notebooks in `notebooks/examples` to be continued
 - lots of tasks added, see `config/tasks/new.yaml`
 - use more recent `torch` and `torchvision` versions

### API changes

### Bug fixes
 - `AA` mode now actually uses `arch.pretrained` (was previously not depending upon it)

### Known Issues
 - performance of segmentation models poorly (opt mode, voc12 task, dice loss)
 - storing paths currently does not transfer between systems (e.g. storing a fim/features/modelstorage on the cluster and transfering it to a local machine)


## 0.4.1 (02/16/2022):
 Some small improvements, mainly to avoid information leakage scenarios in `opt` mode.

### Features
 - new callback `mml.task_optimization.scripts.utils.DropAugmentations` that turns off augmentations as suggested in [here](https://openreview.net/pdf?id=ZcKPWuhG6wy)
 - new augmentation `load_imagenet_aa` loads a (imagenet) pretrained aa augmentation pipeline (found in assets) 
 - `opt` mode also stores all metric progress in `ModelStorage` (via new `MetricCallback`)
 - `infer` mode has new config value `alpha` for weighing in `samples_plus_distance` strategy
 - class weights for losses are scaled more towards `1.0`, which stabilizes training and makes `lr` transferable
 - added `lr_scheduler=step` config option 

### API changes
 - new boolean config value `val_is_test` of mode `opt` determines information leakage of validation split on training 
 - some checks in `opt` mode try to avoid other forms of information leakage as from `EarlyStopping` callback based on `val` loss/metric
 - default number of `trainer.max_epochs` has changed from `200` to `50` in `opt` mode
 - `aa` mode only uses `train` split for training instead of previous `full_train` split

### Bug fixes

### Known Issues
 - performance of segmentation models poorly (opt mode, voc12 task, dice loss)
 - storing paths currently does not transfer between systems (e.g. storing a fim/features/modelstorage on the cluster and transfering it to a local machine)
 - loading time at the beginning becomes a bit long, so probably do both: conditional import on mode (not import for all modes) and split meta info into meta / tuples / folds

## 0.4.0 (02/08/2022):

Since the package setup and arrangement changed, a fresh installation of your environment is necessary! See README! Also 
rename your `local.env` to `personal.env`! Also make sure to set the kaggle credentials during this migration (see `example.env`).

### Features
 - added plotting criteria for the distance methods (e.g. plotting.distance.criteria=task_type)
 - added a lot of tasks, see `mml/configs/tasks/new.yaml` for a full list
 - segmentation encoder uses now same pretrained weights as classification encoder (ATTENTION: Since this uses a feature that has not been released yet atm you have to manually pip uninstall segmentation-models-pytorch and then pip install git+https://github.com/qubvel/segmentation_models.pytorch)
 - added RandAugment as augmentation (see config/augmentations/randaugment)
 - classification tasks now report more metrics
 - installation of the auto-augment environment is now handled as package extra
 - autoalbument fork is now used to support newer versions of timm and segmentation_models_pytorch (together with recent pytorch versions)
 - new mml mode "infer" that uses similarity knowledge to generate pipeline blueprints for target tasks based on previous runs and task similarity
 - new mml mode "crawl" is an alias for optimization mode without a pivot and lower number of epochs
 - `ModelStorage` now wraps trained models and eases reuse of them, see `file_manager.py` for the details
 - it is possible to directly download data via the kaggle api (see `dset_creator.py` for implementation and `covid_xray.py` for example usage)
 - scheduler now acquire a lock to avoid race conditions within the same running folder
 - AutoAugmentation is now able to store pipeline decoupled from preprocessing

### API changes
 - the setup has been extended with `setup.cfg`, `pyproject.toml` and more testing packages, requirements have been updatet
 - new config option: `use_best_params` eases reuse optimal parameters of optuna hpo searches, see README
 - the environments settings file has been renamed from `local.env` to `personal.env` better reflecting its purpose
 - the storing of trained models is now wrapped as ModelStorage (see `mml.core.data_loading.file_manager.py` for details)
 - previously named `quick_tune` routine of optimization mode is renamed to `train_fold`
 - `additional_preparation_instructions` is now `after_preparation_hook` and `additional_finishing_instructions` is now `before_finishing_hook` of `Scheduler`
 - the default preprocessing pipeline has changed to `default.yaml` (instead of `example.yaml`)
 - because auto augmentation pipelines are now composed differently, this breaks backward compatibility with previously generated pipelines
 - TEMP files folder moved from project level to run level
 - `use_best_params` option does not require `hpo=optuna` any more
 - unified optimization direction for metrics is now 'minimize' to ensure compatibility with default optuna setting
 - reusables of the file manager are not longer based on `task.id` but on `task.name`, so they becom independent from dset and task numbering, which could be inconsistent across systems and preprocessings
 - the run folders have now added milliseconds, to avoid racing conditions on the cluster

### Bug fixes
 - refactored endovis18 endoscopic instrument segmentation task
 - fixed plotting after AA mode with preprocessed tasks
 - improved error message if a task was not found
 - if rar backend fails, a better error message is shown
 - an inefficient list comprehension from task_dataset was replaced (caused high latency with big tasks)
 - set pretrained default to true (this caused very inconsistent results for months)
 - `inf` as returned value caused hyperparameter optimization to crash, this is prevented by setting the return value very high instead
 - running `mode=pp` with multiple tasks in parallel on the same dataset caused race condition problems, which is avoided now

### Known Issues
 - performance of segmentation models poorly (opt mode, voc12 task, dice loss)
 - storing paths currently does not transfer between systems (e.g. storing a fim/features/modelstorage on the cluster and transfering it to a local machine)
 - loading time at the beginning becomes a bit long, so probably do both: conditional import on mode (not import for all modes) and split meta info into meta / tuples / folds
 - `ReuseConfig` dataclass still used `tuned` attribute in contrast to hydra config attribute `fc-tuned` 

## 0.3.1 (11/03/2021):

Minor bug fixes and performance improvements.

### Features

### API changes

### Bug fixes
 - Compatible with pytorch-lightning >= 1.4
 - fisher computation speed up
 - fixed CE loss for segmentation
 - feature extraction speed up (if GPU available)
 - fixed incorrect registered tasks enid and glenda
 - fixed "reuse" key error for models with tuned final classifier (fc-tuned)
 - fixed plotting of task similarity to show correct domains

### Known Issues
 - performance of segmentation models poorly (opt mode, voc12 task, dice loss)
 - endovis18 endoscopic instrument segmentation task needs refactoring

## 0.3.0 (10/07/2021):

This version marks the start of collaborative work on MML.

### Features
 - added import functionality ob arbitrary pytorch datasets (see mml.core.data_preparation.DsetCreator.extract_from_pytorch_datasets)
 - added two example datasets for the above
   - SVHN: mml.create_tasks.data_preparation.classification.svhn.py
   - VOC: mml.create_tasks.data_preparation.segmentation.voc.py
 - refactored unit tests (see tests.unit)
 - many more datasets/tasks have been included
 - support of weighted losses for class imbalanced datasets

### API changes
 - set_active_naming is fully handled by BaseScheduler instead of abstract method (see mml.core.scripts.base_scheduler.py)
 - made file manager a singleton class (see mml.core.data_loading.file_manager) so once instantiated by the scheduler it may be called from every other spot
 - transform masks (dataset creator function) now takes a dict transform instead of list

### Bug fixes

### Known Issues
 - Fisher computation as part of FED mode is rather slow
 - Feature extraction as part of e.g., MMD should be sped up as well
 - CE Loss for Segmentation raises error
 - performance of segmentation models poorly (opt mode, voc12 task, dice loss)
 - endovis18 endoscopic instrument segmentation tsak needs refactoring