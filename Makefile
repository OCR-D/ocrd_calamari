# '$(PYTHON)'
PYTHON = python

# '$(PIP_INSTALL)'
PIP_INSTALL = pip install

# '$(GIT_CLONE)'
GIT_CLONE = git clone --depth 1

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install          Install ocrd_calamari"
	@echo "    calamari         Clone calamari repo"
	@echo "    calamari_models  Clone calamari_models repo"
	@echo "    calamari/build   pip install calamari"
	@echo "    deps-test        Install testing python deps via pip"
	@echo "    repo/assets      Clone OCR-D/assets to ./repo/assets"
	@echo "    test/assets      Setup test assets"
	@echo "    assets-clean     Remove symlinks in test/assets"
	@echo "    test             Run unit tests"
	@echo "    coverage         Run unit tests and determine test coverage"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    PYTHON       '$(PYTHON)'"
	@echo "    PIP_INSTALL  '$(PIP_INSTALL)'"
	@echo "    GIT_CLONE    '$(GIT_CLONE)'"

# END-EVAL

# Install ocrd_calamari
install:
	$(PIP_INSTALL) .

# Clone calamari repo
calamari:
	$(GIT_CLONE) https://github.com/chwick/calamari

# Clone calamari_models repo
calamari_models:
	$(GIT_CLONE) https://github.com/chwick/calamari_models

# pip install calamari
calamari/build: calamari calamari_models
	cd calamari && $(PIP_INSTALL) .

#
# Assets and Tests
#

# Install testing python deps via pip
deps-test:
	$(PIP) install -r requirements_test.txt


# Clone OCR-D/assets to ./repo/assets
repo/assets:
	mkdir -p $(dir $@)
	git clone https://github.com/OCR-D/assets "$@"


# Setup test assets
test/assets: repo/assets
	mkdir -p $@
	cp -r -t $@ repo/assets/data/*

# Remove symlinks in test/assets
assets-clean:
	rm -rf test/assets

# Run unit tests
test: test/assets calamari_models
	# declare -p HTTP_PROXY
	$(PYTHON) -m pytest --continue-on-collection-errors test $(PYTEST_ARGS)

# Run unit tests and determine test coverage
coverage:
	coverage erase
	make test PYTHON="coverage run"
	coverage report
	coverage html

.PHONY: assets-clean test
