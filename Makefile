PYTHON ?= python3
APP ?= server.app:app
HOST ?= 0.0.0.0
PORT ?= 7860
SPACE_URL ?= https://aaksh5985-focus-ai.hf.space

.PHONY: run inference local-check space-check

run:
	uvicorn $(APP) --host $(HOST) --port $(PORT)

inference:
	$(PYTHON) inference.py

local-check:
	$(PYTHON) scripts/check_core_artifact.py

space-check:
	$(PYTHON) scripts/smoke_test_space.py --base-url $(SPACE_URL)

quality:
	$(PYTHON) scripts/run_quality_gate.py

pytest:
	pytest -q
