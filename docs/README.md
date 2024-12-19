# MML Docs

We are using Sphinx to build the documentation. The API is documented with the help of 
[autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) and automatically pulls the docstrings 
from all [listed modules](source/api/overview.rst). The CLI interaction is documented within the `.yaml` files of the 
`mml.configs` folder through [autoyaml](https://github.com/Jakski/sphinxcontrib-autoyaml) and referenced in the 
[cli subfolder](source/cli/overview.rst). The gallery of examples is created through 
[MyST-NB](https://github.com/executablebooks/MyST-NB) and the jupyter notebooks in `source/notebooks`.

See following example shows the documentation of a function:

```python
from typing import Optional
from pathlib import Path


def do_something(arg1: Path, kwarg1: Optional[float] = None) -> Path:
    """
    One sentence description. Additional information, e.g. to the kwarg1 behaviour.
    
    .. code-block: python

        # Usage example
        arg1 = ...
        output = do_something(arg1)     

    :param arg1: description of arg1
    :param kwarg1: desciption of kwarg1

    :raises FileExistsError: in case arg1 already exists
    :raises ValueError: in case kwarg1 is not positive

    :return: description of return value
    :rtype: ~pathlib.Path
    """
    if kwarg1 and kwarg1 < 1:
        raise ValueError('Kwarg1 must be a positive integer if provided.')
    if arg1.exists():
        raise FileExistsError('Arg1 already exists.')
    ...
```

For documentation of CLI options the following template can be used for documentation:

```yaml
###
# default: False
#  - provide desciption of parameter
#  - reference to class/method in mml, e.g. :meth:`~mml.core.scripts.schedulers.base_scheduler.AbstractBaseScheduler.create_model`
#  - reference external `links <https://some.where.html>`_
#  - see Sphinx documentation for more syntax details
parameter: False
#  inline comment on top-level key parameter, will not be shown in generated docs
key:
  ###
  # default: 0.5
  #  - document as above
  subparameter: 0.5
```


## Building Docs

After any update to the docs, building them locally and visually inspecting the output is required. No errors should be 
reported by `sphinx` during the build-process. Run these commands from within the `docs` directory:

```bash
make clean
make html
```

and open `docs/build/html/index.html` in your browser. After the PR has been sent an automated pipeline will run tests
and build the docs.
