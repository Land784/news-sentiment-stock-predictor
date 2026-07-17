PYTHON=python3

test:
	pytest -q

demo:
	$(PYTHON) src/inference.py

train:
	$(PYTHON) src/train.py

timing:
	$(PYTHON) analysis/timing.py
