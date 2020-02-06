export  # export variables to subshells
PIP_INSTALL = pip3 install
GIT_CLONE = git clone
PYTHON = python3
PYTEST_ARGS = -W 'ignore::DeprecationWarning' -W 'ignore::FutureWarning'

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install          Install ocrd_calamari"
	@echo "    gt4histocr-calamari Get GT4HistOCR Calamari model (from SBB)"
	@echo "    actevedef_718448162 Download example data"
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


# Get GT4HistOCR Calamari model (from SBB)
gt4histocr-calamari:
	mkdir gt4histocr-calamari
	cd gt4histocr-calamari && \
	wget https://qurator-data.de/calamari-models/GT4HistOCR/model.tar.xz && \
	tar xfv model.tar.xz && \
	rm model.tar.xz

# Download example data
actevedef_718448162:
	wget https://qurator-data.de/examples/actevedef_718448162.zip && \
	unzip actevedef_718448162.zip



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
test: test/assets gt4histocr-calamari
	# declare -p HTTP_PROXY
	$(PYTHON) -m pytest --continue-on-collection-errors test $(PYTEST_ARGS)

# Run unit tests and determine test coverage
coverage: test/assets gt4histocr-calamari
	coverage erase
	make test PYTHON="coverage run"
	coverage report
	coverage html

.PHONY: assets-clean test
