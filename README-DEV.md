Testing
-------
In a Python 3 virtualenv:

~~~
pip install -e .
pip install -r requirements-test.txt
make test
~~~

Releasing
---------
* Update `ocrd-tool.json` version
* Update `setup.py` version
* `git commit -m 'v<version>'`
* `git tag -m 'v<version>' 'v<version>'`
* `git push --tags`
* Do a release on GitHub

### Uploading to PyPI
* `rm -rf dist/` or backup if `dist/` exists already
* In the virtualenv: `python setup.py sdist bdist_wheel`
* `twine upload dist/ocrd_calamari-<version>*`
