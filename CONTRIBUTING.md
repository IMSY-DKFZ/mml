# Contributing to MML

First of all: Thank you for your interest in contributing to `MML`! This document provides guidelines and 
instructions for contributing to this project. In case you have any questions, do not hesitate to get in touch with 
the members of the core development team:

[Patrick Godau](patrick.godau@dkfz-heidelberg.de)


<!-- TOC -->
* [Contributing to MML](#contributing-to-mml)
  * [Getting Started](#getting-started)
  * [Development Process](#development-process)
  * [Code Style](#code-style)
  * [Testing](#testing)
  * [Pull Request Process](#pull-request-process)
  * [License](#license)
  * [Contribution review and integration](#contribution-review-and-integration)
  * [List of Contributors](#list-of-contributors)
<!-- TOC -->

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a virtual environment
4. Install ``mml` in editable mode including the development dependencies:
   ```bash
   pip install -e ".[dev,docs]"
   ```

## Development Process

1. Open an issue and discuss the strategy on how to tackle it
2. Create a new branch for your feature (`feature/`) or bugfix (`fix/`), add the number of the issue (e.g. `feature/123`)
   ```bash
   git checkout -b feature/123
   ```
2. Make your changes, following our coding standards
3. Add tests for any new functionality
4. Run the test suite:
   ```bash
   pytest
   ```
5. Update documentation as needed (see [`docs`](docs/README.md))
6. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

## Code Style

We follow these coding standards:

 * [PEP 8](https://peps.python.org/pep-0008/) - Python style guide
 * [type hints](https://peps.python.org/pep-0484/) for function arguments and return values
 * document functions and classes using `Sphinx` [docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)
 * maximum line length of 120 characters (set in [`pyproject.toml`](pyproject.toml))

We use the following tools for code quality:

 * `ruff` for code formatting
 * `isort` for import sorting
 * `pylint` for overall quality

Run the full suite of checks with:

```commandline
ruff check
ruff format
isort .
pylint src/mml --max-line-length 120
```

## Testing

Add positive and negative test cases. Mock dependencies appropriately. The `mml.testing` package provides a set of 
`pytest.fixtures` you can leverage even when writing tests for plugins.

## Pull Request Process

 * update the [CHANGELOG.md](CHANGELOG.md) following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format
 * make sure all tests pass and code quality checks succeed
 * check documentation, including docstrings of added / changed classes and functions
 * submit a pull request with a clear description of the changes, for this your pull request should:
   * have a clear, descriptive title
   * reference any related issues
   * include a summary of changes
   * note any backward compatibility breaking changes

## License

By contributing, you agree that your contributions will be licensed under the project's license (MIT). 
All added or edited code shall be the own original work of the particular contributor. If you use some third-party 
implementation, all such blocks/functions/modules shall be properly referred and if possible also agreed by code’s 
author. For example - "This code is inspired from http://...". In case you are adding new dependencies, make sure that 
they are compatible with the actual license (i.e. dependencies should be at least as permissive as 
the MIT license).

## Contribution review and integration
To ensure correctness and high quality of the submitted code, each contribution will be checked by the CI pipeline 
and reviewed by a member of the core development team regarding among others the following aspects:
- The code is correct and implements the [described feature / fixes the described issue](#getting-started)
- The code follows our [coding style](#code-style)
- The code is [documented appropriately](docs/README.md)
- The code is covered by sensible [unit tests](#testing) that pass upon submission
- The contribution does not lead to side effects in other parts of the toolkit (e.g. failing tests)
Once the reviewer is content with the contribution, the changes will be integrated into the code base.

## List of Contributors

In the following table we list the people that have contributed to the MML toolkit.

Main author (>99%):

- [Patrick Godau](https://www.dkfz.de/en/imsy/team/people/Patrick_Scholz.html)

Other contributors:

- [Akriti Srivastava](https://de.linkedin.com/in/akriti-srivastava-76041a120)
- [Leon Mayer](https://www.dkfz.de/en/imsy/team/people/Leon_Mayer.html)
- [Dominik Michael](https://www.dkfz.de/en/imsy/team/people/Dominik_Michael.html)
- [Piotr Kalinowski](https://www.dkfz.de/en/imsy/team/people/Piotr_Kalinowski.html)
- [Amine Yamlahi](https://www.dkfz.de/en/imsy/team/people/Amine_ElYamlahi.html)