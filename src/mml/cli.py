# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import os
import shutil
import sys
from pathlib import Path

from mml.__main__ import wrapped_mml


def main() -> None:
    """
    Command line convenience. Allows after setup to call 'mml CONFIG_OPTIONS' instead of 'python -m mml CONFIG_OPTIONS'
    """
    wrapped_mml()


def copy_mml_env_file() -> None:
    """
    This CLI command sets up an `mml.env` file at your current location, based on the `example.env` file shipped with
    `mml`.
    """
    if os.getenv("MML_ENV_PATH", None):
        raise RuntimeError(
            f"Found env variable 'MML_ENV_PATH', has mml already been configured? If you want to "
            f"redo mml env init, make sure MML_ENV_PATH is unset (current "
            f"value={os.getenv('MML_ENV_PATH')})."
        )
    template_env = Path(__file__).parent / "template.env"
    default_env_path = template_env.parent / "mml.env"
    if default_env_path.exists():
        raise RuntimeError(
            f"Found existing mml.env file at default location {default_env_path}. Please remove to "
            f"redo mml env init."
        )
    # copy and rename template
    destination = Path(os.getcwd()) / "mml.env"
    if destination.exists():
        raise RuntimeError(
            f"Found existing mml.env file at current location {destination}. Please remove to " f"redo mml env init."
        )
    shutil.copyfile(src=template_env, dst=destination)
    # report
    print(f"Created mml.env file successfully. Path={destination}")
    print("Your next steps:")
    print("  - modify at least MML_DATA_PATH, MML_RESULTS_PATH and MML_LOCAL_WORKERS accordingly")
    print(f"  - set MML_ENV_PATH variable in your environment (conda env config vars set MML_ENV_PATH={destination})")
    print("  - if you created the file somewhere version controlled, make sure to gitignore this confidential file!")


def copy_mml_configs() -> None:
    """
    This is a CLI command that sets up mml configs outside the mml package, the basic idea is to copy the default
    configs to a location of desire and link that location in the "mml.env".
    """
    destination = Path(os.getcwd()) / "configs"
    source = Path(__file__).parent / "configs"
    print("Copying MML configs. This copies the default configs to your current working dir.")
    if destination.exists():
        print(f"Folder >configs< already exits at {destination}. Remove any artefacts and restart.")
        sys.exit(1)
    print(f"Are you sure to create a folder 'configs' at {destination}? [Y/n]")
    answer = input()
    if answer.lower() not in ["y", "n", ""]:
        print("Invalid input, will cancel initialisation.")
        sys.exit(1)
    if answer.lower() == "n":
        print("MML config copy canceled.")
        sys.exit(0)
    # copy configs folder
    shutil.copytree(src=source, dst=destination)
    print("Configs folder created successfully!")
    # determine env file
    template_env = Path(__file__).parent / "example.env"
    assert template_env.exists(), "Template for env variables not found!"
    if os.getenv("MML_ENV_PATH", None):
        env_path = Path(os.getenv("MML_ENV_PATH"))
    else:
        env_path = template_env.parent / "mml.env"
    if not env_path.exists():
        print("No mml.env file found, will create from template.")
        shutil.copyfile(src=template_env, dst=env_path)
    # read in config env file
    with open(env_path, "r") as f:
        lines = f.readlines()
    # find line to modify
    prefix = "export MML_CONFIGS_PATH"
    try:
        line_idx = [line.startswith(prefix) for line in lines].index(True)
    except ValueError:
        print(f"No line in {env_path.name} starts with {prefix}. Will add a line, make sure no mess up happened!")
        lines.append(prefix)
        line_idx = len(lines)
    # modify line
    lines[line_idx] = (
        f"{prefix}={destination.absolute()} # AUTO GENERATED FROM MML-COPY-CONF set back to "
        f"DEFAULT_CONF_PATH to fall back to original configs location\n"
    )
    with open(env_path, "w") as f:
        f.writelines(lines)
    print(f"Modified {env_path.name} successfully. Please test setup via calling <mml> from the CLI. ")
