MODULE := estimator
TEST := test

run:
	@python -m $(MODULE)

test:
	@python -m unittest discover $(TEST)

lint:
	@pylint **/*.py

