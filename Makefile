GIT_CLONE = git clone --depth 1
calamari:
	$(GIT_CLONE) https://github.com/chwick/calamari

calamari_models:
	$(GIT_CLONE) https://github.com/chwick/calamari_models

calamari/build: calamari calamari_models
	cd calamari &&\
		pip install -r requirements.txt ;\
		python setup.py install
