
.PHONY: setup run clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	python scripts/run_pipeline.py --config configs/experiment.yaml

clean:
	rm -rf runs
