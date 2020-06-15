MODULE := estimator
TEST := test

help:
	@echo "usage:"
	@echo "  install"
	@echo "    installs dependencies for this project"
	@echo "  run"
	@echo "    runs estimator on manufactured data"
	@echo "  run DATASET=<N>"
	@echo "    runs estimator on nth dataset where <N> is in {1..3}"
	@echo "  test"
	@echo "    runs suite of unit tests"
	@echo "  lint"
	@echo "    runs linting on all source python files"

install:
	@./setup.py install

run:
	@python -m $(MODULE) -d $(DATASET)

test:
	@python -m unittest discover $(TEST)

lint:
	@pylint --rcfile=.pylintrc -d import-error $(MODULE) 

