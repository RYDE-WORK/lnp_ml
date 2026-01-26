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

## Process dataset for final training (interim -> processed/final, train:val=9:1, no test)
.PHONY: data_final
data_final: requirements
	$(PYTHON_INTERPRETER) scripts/process_data_final.py

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

## Final training using all data (train:val=9:1, no test set), with pretrained weights
.PHONY: train_final
train_final: requirements
	$(PYTHON_INTERPRETER) -m lnp_ml.modeling.train \
		--train-path data/processed/final/train.parquet \
		--val-path data/processed/final/val.parquet \
		--output-dir models/final \
		--init-from-pretrain models/pretrain_delivery.pt \
		$(FREEZE_FLAG) $(MPNN_FLAG) $(DEVICE_FLAG)

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

## Formulation optimization: find optimal LNP formulation for target organ
## Usage: make optimize SMILES="CC(C)..." ORGAN=liver
.PHONY: optimize
optimize: requirements
	$(PYTHON_INTERPRETER) -m app.optimize --smiles "$(SMILES)" --organ $(ORGAN) $(DEVICE_FLAG)

## Start FastAPI backend server (port 8000)
.PHONY: api
api: requirements
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

## Start Streamlit frontend app (port 8501)
.PHONY: webapp
webapp: requirements
	streamlit run app/app.py --server.port 8501

## Start both API and webapp (run in separate terminals)
.PHONY: serve
serve:
	@echo "请在两个终端分别运行:"
	@echo "  终端 1: make api"
	@echo "  终端 2: make webapp"
	@echo ""
	@echo "然后访问: http://localhost:8501"


#################################################################################
# DOCKER COMMANDS                                                               #
#################################################################################

## Build Docker images
.PHONY: docker-build
docker-build:
	docker compose build

## Start all services with Docker Compose
.PHONY: docker-up
docker-up:
	docker compose up -d

## Stop all Docker services
.PHONY: docker-down
docker-down:
	docker compose down

## View Docker logs
.PHONY: docker-logs
docker-logs:
	docker compose logs -f

## Build and start all services
.PHONY: docker-serve
docker-serve: docker-build docker-up
	@echo ""
	@echo "🚀 服务已启动!"
	@echo "   - API:       http://localhost:8000"
	@echo "   - Web 应用:  http://localhost:8501"
	@echo ""
	@echo "查看日志: make docker-logs"
	@echo "停止服务: make docker-down"

## Clean Docker resources (images, volumes, etc.)
.PHONY: docker-clean
docker-clean:
	docker compose down -v --rmi local
	docker system prune -f


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
