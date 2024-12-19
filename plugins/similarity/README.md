# MML task similarity plugin

This plugin provides a wide range of modes to compute task similarities.

# Install

```commandline
pip install mml-similarity
```

# Usage

The new mode `similarity` offers to compute task distances. For example to compute the `Fisher embedding distance` 
call

```commandline
mml similarity distance=fed ...
```

Task distances can be leveraged by other plugins - e.g. `mml-suggest`.