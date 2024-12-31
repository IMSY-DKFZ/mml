# MML Suggest plugin

This plugin provides a method to compile pipeline blueprints for new tasks. 

# Install

> Note: this plugin depends on another plugin (mml-similarity) which is installed automatically - though 
> this has no side effects.

```commandline
pip install mml-suggest
```


# Usage

To leverage existing models and task distances these must be reused:

```commandline
mml suggest pivot.name=my_target_task +reuse.fed=my_distance_project reuse.models=[some_proj,another_proj,...]
```

This will compile a pipeline blueprint for `my_target_task` based on the `fed` task similarities from 
`my_distance_project` leveraging all models previously trained in the specified projects.

To use the blueprint call 

```commandline
mml train pivot.name=my_target_task reuse.blueprint=my_blueprint_proj#the_blueprint_number mode.use_blueprint=true
```