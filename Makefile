MODULE := estimator
TEST := test

run:
	@python -m $(MODULE) -d $(DATASET)

test:
	@python -m unittest discover $(TEST)

lint:
	@pylint $(MODULE) 

