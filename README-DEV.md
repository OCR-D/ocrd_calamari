Testing
-------
In a Python 3 virtualenv:

~~~
pip install -e .
pip install -r requirements-test.txt
make test
~~~

Release
-------
* Update ocrd-tool.json version
* Update setup.py version
* git commit -m 'v<version>'
* git tag -m 'v<version>' 'v<version>'
* git push --tags

PyPI:
* python sdist bdist_wheel
* twine upload dist/ocrd_calamari-<version>*
