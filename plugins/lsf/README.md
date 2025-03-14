# MML LSF plugin

This plugin provides LSF cluster support on the specific setting of the DKFZ GPU cluster.

# Install

```commandline
pip install mml-lsf
```

If you want to use the submission features of `mml_lsf.runner.LSFJobRunner` you need to set up the following 
(not required by any of the other features of this plugin):

  * install sshpass for providing ssh with password
      ```commandline
      sudo apt install sshpass
      ```
  * set the following variables in your mml.env (alternatively provide them manually to `mml_lsf.runner.LSFJobRunner`)
    * `export MML_AD_USER=...`
    * `export MML_CLUSTER_HOST=...`  (a LSF submission host)
    * `export MML_CLUSTER_WORKER=...`  (a cluster worker node)

# Usage

First and foremost it automatically ensures the number of workers used by MML to be conforming to 
the node the job will be executed (see `mml_lsf.workers`). In addition, it provides a suitable implementation 
for job planning on the LSF cluster, taking care of all necessary prefixes to the CLI (see 
`mml_lsf.requirements`). Finally, it offers the `LSFJobRunner` to automatically submit job. Alternatively it is also 
possible to submit via pre-rendering into a local file and ssh file tunneling.

## Usage with `sshpass'

```python
from mml.core.scripts.utils import load_env
from mml_lsf.requirements import LSFSubmissionRequirements
from mml_lsf.runner import LSFJobRunner
from mml.interactive.planning import MMLJobDescription

# make sure to load mml.env variables
load_env()  # if within a jupyter notebook, instead invoke mml.interactive.init()
# setup job requirements
reqs = LSFSubmissionRequirements(
    num_gpus=1, 
    vram_per_gpu=11.0, 
    queue='gpu-lowprio',
    mail='something@dkfz-heidelberg.de',  # optional
    script_name='mml.sh',  # name of my runner script to load CUDA, conda env, etc and finally invoke mml
    job_group='/USERNAME/JOB_GROUP_NAME',   # optional, used e.g. to limit max number of jobs 
    interactive=True  # optional, if True realtime updates are printed to terminal
    )
# setup runner, will prompt for password once
runner = LSFJobRunner() 
job = MMLJobDescription(prefix_req=reqs, mode='info', config_options={})  # simple job "mml info"
job.run(runner=runner)  # will submit job (no password prompt)
job2 = MMLJobDescription(prefix_req=reqs, mode='train', config_options={})  # another job "mml info"
job2.run(runner=runner) # will submit job (no password prompt)
```

## Usage with file tunneling

```python
from mml_lsf.requirements import LSFSubmissionRequirements
from mml.interactive.planning import MMLJobDescription, write_out_commands

# setup job requirements
reqs = LSFSubmissionRequirements(
    num_gpus=1, 
    vram_per_gpu=11.0, 
    queue='gpu-lowprio',
    mail='something@dkfz-heidelberg.de',  # optional
    script_name='mml.sh',  # name of my runner script to load CUDA, conda env, etc and finally invoke mml
    job_group='/USERNAME/JOB_GROUP_NAME',   # optional, used e.g. to limit max number of jobs 
    interactive=False  # setting True is not recommended for batched submission
    )

# create batch of cmds
cmds = list()
# cmd 1 some dummy task
cmds.append(MMLJobDescription(prefix_req=reqs, mode='train', config_options={'tasks': 'fake', 'proj': 'dummy'}))
# cmd 2 another dummy task
cmds.append(MMLJobDescription(prefix_req=reqs, mode='train', config_options={'tasks': 'fake', 'proj': 'dummy'}))
# now write
write_out_commands(cmd_list=cmds, name='exp1')
# this creates a 'exp1.txt' at current working directory
```

Now submit these jobs via:

```commandline
ssh AD_USER@SUBMISSION_HOST 'bash -s' < /path/to/exp1.txt
```



