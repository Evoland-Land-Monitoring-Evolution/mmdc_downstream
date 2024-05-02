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


test_architecture:
	$(CONDA) && pytest -vv test/test_architecture.py


test_datamodule:
	$(CONDA) && pytest -vv test/test_mmdc_datamodule.py

test_stats:
	$(CONDA) && pytest -vv test/test_stats.py

test_gends:
	$(CONDA) && pytest -vv test/test_generate_mmdc_ds.py

test_pix2pix:
	$(CONDA) && pytest -vv test/test_pix2pix.py

test_translation:
	$(CONDA) && pytest -vv test/test_translation_s1_s2.py

test_modtranslation:
	$(CONDA) && pytest -vv -x test/test_modular_translation.py

test_full:
	$(CONDA) && pytest -vv -x test/test_mmdc_full_module.py

test_no_translation:
	$(CONDA) && pytest -vv test/ -k 'not translation'

test_no_PIL:
	$(CONDA) && pytest -vv test/ -k 'not callbacks and not visu'

test_mask_loss:
	$(CONDA) && pytest -vv test/test_masked_losses.py

test_despeckle:
	$(CONDA) && pytest -vv test/test_filters.py

test_iota2:
	$(CONDA) && pytest -vv test/test_iota2.py

test_inference:
	$(CONDA) && pytest -vv test/test_inference.py


PYLINT_IGNORED = "miou.py, weight_init.py"
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
