export  # export variables to subshells
PIP_INSTALL = pip3 install
GIT_CLONE = git clone
PYTHON = python3
PYTEST_ARGS = -W 'ignore::DeprecationWarning' -W 'ignore::FutureWarning' -vv
# not usable with Calamari 2 ATM - see Calamari#362
#MODEL = qurator-gt4histocr-1.0 # cannot be migrated to Calamari 2
#MODEL = deep3_fraktur19 # too large for CI
MODEL = fraktur_19th_century
export MODEL # needed for pytest model selection
EXAMPLE = actevedef_718448162.first-page+binarization+segmentation

DOCKER_TAG = 'ocrd/calamari'
DOCKER_BASE_IMAGE = docker.io/ocrd/core-cuda-tf2:v3.1.1

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install          Install ocrd_calamari"
	@echo "    install-dev      Install in editable mode"
	@echo "    build            Build source and binary distribution"
	@echo "    docker           Build Docker image"
	@echo "    $(MODEL)         Get Calamari model for tests"
	@echo "    example          Download example data"
	@echo "    deps-test        Install testing python deps via pip"
	@echo "    repo/assets      Clone OCR-D/assets to ./repo/assets"
	@echo "    test/assets      Setup test assets"
	@echo "    assets-clean     Remove symlinks in test/assets"
	@echo "    test             Run unit tests"
	@echo "    coverage         Run unit tests and determine test coverage"
	@echo "    docker           Build Docker image"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    PYTHON            '$(PYTHON)'"
	@echo "    PIP_INSTALL       '$(PIP_INSTALL)'"
	@echo "    GIT_CLONE         '$(GIT_CLONE)'"
	@echo "    MODEL             '$(MODEL)'"
	@echo "    DOCKER_TAG        '$(DOCKER_TAG)'"
	@echo "    DOCKER_BASE_IMAGE '$(DOCKER_BASE_IMAGE)'"

# END-EVAL

# Install ocrd_calamari
install:
	$(PIP_INSTALL) .

install-dev:
	$(PIP_INSTALL) -e .

build:
	$(PIP_INSTALL) build
	$(PYTHON) -m build .

# Build docker image
docker:
	docker build \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG) .

# Get GT4HistOCR Calamari model (from SBB)

$(MODEL):
	ocrd resmgr download ocrd-calamari-recognize $@

# Download example data (for the README)
example: $(EXAMPLE)

$(EXAMPLE):
	wget -c https://qurator-data.de/examples/$(EXAMPLE).zip -O $(EXAMPLE).zip.tmp
	mv $(EXAMPLE).zip.tmp $(EXAMPLE).zip
	unzip $(EXAMPLE).zip
	rm $(EXAMPLE).zip



#
# Assets and Tests
#

# Install testing python deps via pip
deps-test:
	$(PIP_INSTALL) -r requirements-dev.txt

deps-test-ubuntu: deps-test
	apt-get install -y make git curl wget imagemagick


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
test: test/assets $(MODEL)
	# declare -p HTTP_PROXY
	$(PYTHON) -m pytest --continue-on-collection-errors --durations=0 test $(PYTEST_ARGS)

# Run unit tests and determine test coverage
coverage: test/assets $(MODEL)
	coverage erase
	make test PYTHON="coverage run"
	coverage combine
	coverage report
	coverage html

.PHONY: install install-dev build docker assets-clean deps-test test coverage $(MODEL) example
