# MML drive plugin

This plugin provides support for fast data installation via a shared network drives. It extends the create scheduler 
to look for respective downloaded files there first. This might be useful if a team works jointly with `mml` and you 
want to avoid redundant downloading of raw datasets.  

# Install

After installing this plugin via `pip` you need to mount a network drive. On that network drive there needs to reside 
the following file structure: `MedicalMetaLearner/DOWNLOADS`. You may copy contents of your local 
`MML_DATA_PATH/DOWNLOADS` into that folder to provide the downloaded files to others. Every user that wants to download 
is required to add `export MML_NW_DRIVE=path/to/drive/root` to their `mml.env` file.

# Usage

Just use `mml create` as usual. If the network drive offers a benefit, `mml` will download data from there.