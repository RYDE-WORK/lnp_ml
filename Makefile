#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = lnp-ml
PYTHON_VERSION = 3.8
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pixi install




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	@echo ">>> Pixi environment will be created when running 'make requirements'"
	
	@echo ">>> Activate with:\npixi shell"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Clean raw data (raw -> interim)
.PHONY: clean_data
clean_data: requirements
	$(PYTHON_INTERPRETER) scripts/data_cleaning.py

## Process dataset (interim -> processed)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) scripts/process_data.py

## Process external data for pretrain (external -> processed)
.PHONY: data_pretrain
data_pretrain: requirements
	$(PYTHON_INTERPRETER) scripts/process_external.py

## Process CV data for cross-validation pretrain (external/all_amine_split_for_LiON -> processed/cv)
.PHONY: data_pretrain_cv
data_pretrain_cv: requirements
	$(PYTHON_INTERPRETER) scripts/process_external_cv.py

## Process internal data with CV splitting (interim -> processed/cv)
## Use SCAFFOLD_SPLIT=1 to enable amine-based scaffold splitting (default: random shuffle)
SCAFFOLD_SPLIT_FLAG = $(if $(filter 1,$(SCAFFOLD_SPLIT)),--scaffold-split,)

.PHONY: data_cv
data_cv: requirements
	$(PYTHON_INTERPRETER) scripts/process_data_cv.py $(SCAFFOLD_SPLIT_FLAG)

# MPNN 支持：使用 USE_MPNN=1 启用 MPNN encoder
# 例如：make pretrain USE_MPNN=1
MPNN_FLAG = $(if $(USE_MPNN),--use-mpnn,)

# Backbone 冻结：使用 FREEZE_BACKBONE=1 冻结 backbone，只训练 heads
# 例如：make finetune FREEZE_BACKBONE=1
FREEZE_FLAG = $(if $(FREEZE_BACKBONE),--freeze-backbone,)

# 设备选择：使用 DEVICE=xxx 指定设备
# 例如：make train DEVICE=cuda:0 或 make test_cv DEVICE=mps
DEVICE_FLAG = $(if $(DEVICE),--device $(DEVICE),)

## Pretrain on external data (delivery only)
.PHONY: pretrain
pretrain: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.pretrain main $(MPNN_FLAG) $(DEVICE_FLAG)

## Evaluate pretrain model (delivery metrics)
.PHONY: test_pretrain
test_pretrain: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.pretrain test $(MPNN_FLAG) $(DEVICE_FLAG)

## Pretrain with cross-validation (5-fold)
.PHONY: pretrain_cv
pretrain_cv: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.pretrain_cv main $(MPNN_FLAG) $(DEVICE_FLAG)

## Evaluate CV pretrain models on test sets (auto-detects MPNN from checkpoint)
.PHONY: test_pretrain_cv
test_pretrain_cv: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.pretrain_cv test $(DEVICE_FLAG)

## Train model (multi-task, from scratch)
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.train $(MPNN_FLAG) $(DEVICE_FLAG)

## Finetune from pretrained checkpoint (use FREEZE_BACKBONE=1 to freeze backbone)
.PHONY: finetune
finetune: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.train --init-from-pretrain models/pretrain_delivery.pt $(FREEZE_FLAG) $(MPNN_FLAG) $(DEVICE_FLAG)

## Finetune with cross-validation on internal data (5-fold, amine-based split) with pretrained weights
.PHONY: finetune_cv
finetune_cv: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.train_cv main --init-from-pretrain models/pretrain_delivery.pt $(FREEZE_FLAG) $(MPNN_FLAG) $(DEVICE_FLAG)

## Train with cross-validation on internal data only (5-fold, amine-based split)
.PHONY: train_cv
train_cv: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.train_cv main $(FREEZE_FLAG) $(MPNN_FLAG) $(DEVICE_FLAG)


## Evaluate CV finetuned models on test sets (auto-detects MPNN from checkpoint)
.PHONY: test_cv
test_cv: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.train_cv test $(DEVICE_FLAG)

## Train with hyperparameter tuning
.PHONY: tune
tune: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.train --tune $(MPNN_FLAG) $(DEVICE_FLAG)

## Run predictions
.PHONY: predict
predict: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.predict $(DEVICE_FLAG)

## Test model on test set (with detailed metrics, auto-detects MPNN from checkpoint)
.PHONY: test
test: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.predict test $(DEVICE_FLAG)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
