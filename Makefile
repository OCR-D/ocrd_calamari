GIT_CLONE = git clone --depth 1

# Docker tag
DOCKER_TAG = ocrd/calamari

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    calamari         git clone calamari"
	@echo "    calamari_models  git clone calamari_models"
	@echo "    calamari/build   Install calamari"
	@echo "    docker           Build docker image"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    DOCKER_TAG  Docker tag"

# END-EVAL

# git clone calamari
calamari:
	$(GIT_CLONE) https://github.com/chwick/calamari

# git clone calamari_models
calamari_models:
	$(GIT_CLONE) https://github.com/chwick/calamari_models

# Install calamari
calamari/build: calamari calamari_models
	cd calamari &&\
		pip install -r requirements.txt ;\
		python setup.py install

# Build docker image
docker:
	docker build -t '$(DOCKER_TAG)' .
