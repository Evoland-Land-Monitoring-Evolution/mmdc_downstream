CONDA_ENV = mmdc_safe_environment
CONDA_SRC = echo ""
MODULE_PURGE = echo ""
CONDA_ACT = echo ""
ifneq ($(wildcard /work/CESBIO/*),)
	MODULE_PURGE:=module purge
	CONDA_SRC:=module load conda/22.11.1
	CONDA_ACT:=conda activate $(CONDA_ENV)
endif
CONDA = $(MODULE_PURGE) && $(CONDA_SRC) && $(CONDA_ACT)
PYPATH = PYTHONPATH=./src/:${PYTHONPATH}

all: test

#####################
# Environment setup #
#####################
.PHONY:
build_conda:
	bash activate-conda-env.sh $(CONDA_ENV)

#######################
# Testing and linting #
#######################
.PHONY: check
check: test pylint mypy

.PHONY: test
test:
	$(CONDA) && python -m pytest -vv test/

test_no_slow:
	$(CONDA) && pytest -vv -m "not slow" test/

test_no_hal:
	$(CONDA) && pytest -vv -m "not hal" test/

test_no_hal_one_fail:
	$(CONDA) && pytest -vv -x -m "not hal" test/

test_one_fail:
	$(CONDA) && pytest -vv -x test/

test_one_fail_no_slow:
	$(CONDA) && pytest -vv -m "not slow" -x test/



PYLINT_IGNORED = "write_csv.py,lai_regression_compute_bins.py,lai_snap.py"
#.PHONY:
pylint:
	$(CONDA) && pylint --ignore=$(PYLINT_IGNORED) src/

#.PHONY:
ruff:
	$(CONDA) && ruff check . --format pylint

#.PHONY:
mypy:
	$(CONDA) && mypy --exclude pix2pix --exclude residual src/

#.PHONY:
pyupgrade:
	$(CONDA) && find ./src/mmdc_downstream/ -type f -name "*.py" -print |xargs pyupgrade --py310-plus

#.PHONY:
autowalrus:
	$(CONDA) && find ./src/mmdc_downstream/ -type f -name "*.py" -print |xargs auto-walrus

#.PHONY:
refurb:
	$(CONDA) && refurb --quiet src/

#.PHONY:
lint: pylint mypy refurb

#.PHONY:
#doc:
#	$(#CONDA) && sphinx-build docs docs/_build

#tox:
#	$(#CONDA) && tox
