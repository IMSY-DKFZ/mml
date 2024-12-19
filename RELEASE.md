### Steps to take before releasing a new version

#### Prerequisites

- Has the `CHANGELOG.md` been updated?
- Is the `README.md` up to date?
- Is the documentation up to date?

#### Quality control

Run the following in order. Treat upcoming issues. 
This still happens on individual develop branches!

- `ruff check` + `ruff format`
- `mypy src`  (currently not necessary)
- `pylint src/mml --max-line-length 120`
- `isort .`
- `pytest`

Commit changes!

#### Check docs

- `cd docs`
- `make clean`
- `make html`
- Open `docs/build/index.html` and check any changes 
- Finally, create a  merge request to `dev`

#### Dev branch

- Continue once all individual branches have been merged into the main `dev` branch
- Set release version in `CHANGELOG.md`, `CITATION.cff` and `src.mml.__init__.py`
- Optionally set version of plugins (and `mml-core` dependency)
- Commit (-m "bump version(s)")
- Next we need to update license headers, be careful to with any potential newly added config files as a fresh license 
header may mess with hydra's `@package` header, directive!

```commandline
find . -type f \( -iname \*.py -o -iname \*.ini -o -iname \*.toml -o -iname \*.yaml \) -not -path "./.tox/*" -exec add-license-header --license-file license_header.template --create-year 2024 --single-year-if-same {} \;
```
- Commit (-m "update license header")
- Push and wait for pipeline to succeed

#### Master branch

- Create merge request from `dev` to `master`
- Wait for pipeline to succeed
- Trigger required manual release jobs:
  - release documentation
  - release plugins
- Checkout `master`
- Create tag (naming: major.minor.patch)
- Push tag