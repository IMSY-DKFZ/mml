# MML Plugin template

You may use this template to design new MML plugins. Follow the steps to make sure maximum compatibility:

 - Create a copy of the template, rename according to the central feature the plugin offers, by convention 
   "mml_" + this single word will also be the name of your package
 - Use that name to rename ROOT/src/mml_TODO
 - Within ROOT/setup.cfg you need to replace mml_TODO three times, for the entries "name" and "version" at the very 
   beginning and once within the "entry_points" section. Sometimes the convention is to use "-" instead of "_".
 - You may provide a useful description in the ROOT/setup.cfg under "description"
 - If you want to provide new configs just place them within ROOT/src/mml_TODO/configs in the same way as you would 
   place them in the mml-core configs folder (e.g. create a folder "mode" and therein a "fancy.yaml" file to call
   a FancyScheduler you implemented in this plugin.)
 - If you provide any config files, make sure to adapt the SearchPathPlugin in ROOT/src/mml_TODO/activate.py, otherwise 
   you must remove it.
 - Place any logic to load your plugins features into MML runtime within ROOT/src/mml_TODO/activate.py, e.g. taskcreator 
   imports to create registry entries or any modification of MML components.
 - Lastly do not forget to provide a meaningful README.md, you can find a small template below.

# MML ??? plugin

This plugin provides ???

# Install

TODO

# Usage

TODO