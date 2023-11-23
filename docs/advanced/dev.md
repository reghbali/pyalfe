# Development

## Pull request
To create a pull request fork the repository, create a branch, commit your
changes to the branch and then create a pull request into the main.

## Test coverage
All new functionalities and modification or existing ones should be covered by unittests.

## Style
We use numpy docstring format, ruff for linting, and black for code formatting.
The pre-commit hook runs ruff and black for before each commit. To enable it
first install pre-commit:

```bash
pip install pre-commit
```
and then run

```bash
pre-commit install # or
pre-commit run -a
```

## Building docs
To build docs run:

```bash
jupyter book build docs
```
