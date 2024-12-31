mml-suggest plugin
====================

The infer mode compiles a strategy to transfer knowledge from other tasks to a specific target task. Hence it is
necessary to set `pivot.name` as one of the tasks provided in suggest mode. Furthermore it is
recommended to include a variety of closely related tasks to the task list - leave it empty to let the scheduler
determine related tasks. Prediction of task similarity must have been performed before and
previous model trainings on the related tasks are required. The following series of commands separates
each of the runs into a separate project, which is up to the user to decide. Alternatively all commands may run
upon the same project.

The pipeline can be orchestrated as follows:
 - **predicting** task similarity: `mml similarity distance=fed proj=my_distances`
 - **crawling** information: `mml train proj=my_crawling` (it is recommended to run this often with a variety of setting on the pipeline, e.g. learning rate, model, ... you may also consider hyperparameter optimization on each of the tasks available or gridsearch multiple setups on all tasks to ease the command)
 - **infering** a blueprint pipeline: `mml suggest proj=my_inference reuse.models=my_crawling +reuse.fed=my_distances`
 - **optimizing** on the target task: `mml train proj=my_optimization mode.use_blueprint=true reuse.blueprint=my_inference`