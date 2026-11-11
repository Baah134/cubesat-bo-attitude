# Reproduce everything: `make all` (hours: reruns every optimizer x seed).
# `make figures` rebuilds tables and figures from the committed histories (minutes).
# Windows without make: `python reproduce.py all` runs the same steps.
PY ?= python
RUN = $(PY) -W ignore -m

.PHONY: all test calibrate stageA stageB stageW robust largeslew figures clean-results

all: test stageA stageB stageW robust largeslew figures

test:
	$(PY) -m pytest tests -q

calibrate:
	$(RUN) scripts.calibrate_specs --stage A --n 512

stageA:
	$(RUN) scripts.run_stage --config configs/stageA.yaml
	$(RUN) scripts.summarize_stage --stage A

stageB:
	$(RUN) scripts.run_stage --config configs/stageB.yaml
	$(RUN) scripts.run_stage --config configs/stageB_robust.yaml
	$(RUN) scripts.summarize_stage --stage B

stageW:
	$(RUN) scripts.run_stage --config configs/stageW.yaml
	$(RUN) scripts.summarize_stage --stage W

robust:
	$(RUN) scripts.robustness --stage B --n 200
	$(RUN) scripts.robustness --stage W --n 200

largeslew:
	$(RUN) scripts.large_slew --n 200 --stages B W

figures:
	$(RUN) scripts.summarize_stage --stage A
	$(RUN) scripts.summarize_stage --stage B
	$(RUN) scripts.summarize_stage --stage W
	$(RUN) scripts.sanity_figure --stage A
	$(RUN) scripts.sanity_figure --stage B
	$(RUN) scripts.figures_final

# Histories are skipped when present; delete them to force a full rerun.
clean-results:
	rm -rf results/histories
