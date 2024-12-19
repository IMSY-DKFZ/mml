# MML Data plugin

This plugin provides a wide range of datasets and tasks, ready to be imported.

> In order to create tasks 'mml' might automatically download data. It is your responsibility to take care of the 
> individual licensing regulations of such data and comply with them. See the respective task creators for more details.

## Install

```commandline
pip install mml-data
```

Some of the tasks require the [kaggle API](https://github.com/Kaggle/kaggle-api), which requires some authentication 
to be set up. Based on the documentation this can be done via a `kaggle.json` file at a location depending on your OS, 
or as the MML internal solution add your [credentials](https://github.com/Kaggle/kaggle-api#api-credentials) to the 
`mml.env` file.

```commandline
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
```

## Usage

Tasks can be created with the `create` mode as usual. Furthermore, mml-data provides some convenience functionality 
to show and filter for available tasks. Type `mml-data --help` for more details.

## Outdated URLs

Dataset download links might become deprecated over time. Please report any outdated urls via the issue tracker and we 
will try to fix those.