# Certified Neural Lyapunov Functions - build and reproduce
#
# On this Windows host the bare `python` alias is the broken Store shim, so pass
# the real interpreter explicitly, e.g.
#   make quick PYTHON=C:/Users/sehaj/anaconda3/python.exe
# On Colab or Linux plain `python` works.

PYTHON ?= python
SEEDS  ?= 0 1 2 3 4
export KMP_DUPLICATE_LIB_OK = TRUE

.PHONY: quick full figures paper certtrain clean

## quick: smoke-test every path in a few minutes from a clean clone
quick:
	$(PYTHON) tests/test_dynamics.py
	$(PYTHON) tests/test_verifier_ops.py
	$(PYTHON) experiments/e0_reproduce.py --seed 0 --iters 800
	$(PYTHON) experiments/e1_sampling_gap.py --seed 0
	$(PYTHON) experiments/e4_cegis.py --seed 0 --iters 2
	$(PYTHON) experiments/e2_certify_4a.py --seed 0
	$(PYTHON) experiments/e3_rung1_twobus.py --seed 0
	$(PYTHON) experiments/e7_certified_train.py --seed 0 --steps 60

## full: every experiment across all seeds, then the boundary study
full:
	$(PYTHON) experiments/sweep.py $(SEEDS)
	$(PYTHON) experiments/e5_boundary.py --seed 0
	$(MAKE) certtrain PYTHON=$(PYTHON)

## certtrain: certified training (E7, CT-BaB) across all seeds, the CEGIS
## head-to-head on the identical warm-started network. 2-D gen-5 slice only;
## raise the dimension by editing configs/certified_train.yaml, see SCALING.md.
certtrain:
	$(PYTHON) experiments/e7_certified_train.py --all

## figures: regenerate every figure from results/*.json and the saved networks
figures:
	$(PYTHON) experiments/make_figures.py

## paper: compile the whitepaper PDF from the same figures
paper: figures
	cd paper && latexmk -pdf main.tex

clean:
	cd paper && latexmk -C
