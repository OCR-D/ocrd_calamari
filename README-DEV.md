Testing
-------
In a Python 3 virtualenv:

```
pip install -e .
pip install -r requirements-dev.txt
make test
```

Releasing
---------
* Update `ocrd-tool.json` version (the `setup.py` version is read from this)
* `git add` the `ocrd-tool.json` file and `git commit -m 'v<version>'`
* `git tag -m 'v<version>' 'v<version>'`
* `git push; git push --tags`
* Wait and check if tests on CircleCI are OK
* Do a release on GitHub

### Uploading to PyPI
* `rm -rf dist/` or backup if `dist/` exists already
* In the virtualenv: `python setup.py sdist bdist_wheel`
* `twine upload dist/ocrd_calamari-<version>*`


How to use pre-commit
---------------------

This project optionally uses [pre-commit](https://pre-commit.com) to check commits. To use it:

- Install pre-commit, e.g. `pip install -r requirements-dev.txt`
- Install the repo-local git hooks: `pre-commit install`
