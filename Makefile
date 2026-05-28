# ==============================================================
# HiRes vs LoRes vessel segmentation — 2×2 cross-validation study
#
# Experiments: {hires, lores} × {dice_ce, cldice}
#
# Steps:
#   setup      — symlink original NIfTI files into MIST directory layout
#   analyze    — mist_analyze per dataset (produces config.json)
#   preprocess — mist_preprocess per dataset (shared numpy arrays)
#   train      — 4 mist_train jobs, one per (dataset, loss) combination
#   summary    — aggregate results.csv files into comparison table
#
# Usage:
#   make all             # run full pipeline sequentially
#   make -j4 train       # run all 4 training jobs in parallel (if GPUs allow)
#   make summary         # regenerate comparison table from existing results
#   make clean           # remove results/ and numpy/ (keeps symlinks)
#   make distclean       # also remove datasets/*/raw/ symlink trees
# ==============================================================

# ---- Tuneable parameters ----
NFOLDS  ?= 5
EPOCHS  ?= 300
MODEL   ?= nnunet
BATCH   ?= 2
WORKERS ?= 4

# ---- Docker parameters ----
IMAGE    ?= mist-hireslores
DATA_DIR ?= /rsrch3/ip/dtfuentes/github/oncopigdata

DOCKER_RUN = docker run --rm \
	--device=/dev/nvidia0 \
	--device=/dev/nvidiactl \
	--device=/dev/nvidia-uvm \
	--device=/dev/nvidia-uvm-tools \
	-u $$(id -u):$$(id -g) \
	-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
	-v $(PWD):/workspace \
	-v $(DATA_DIR):$(DATA_DIR):ro \
	$(IMAGE)

MIST_PARAMS = NFOLDS=$(NFOLDS) EPOCHS=$(EPOCHS) MODEL=$(MODEL) BATCH=$(BATCH) WORKERS=$(WORKERS)

# ---- Experiment matrix ----
DATASETS := hires lores
LOSSES   := dice_ce cldice

.PHONY: all setup analyze preprocess experiments train summary clean distclean \
        docker-build docker-run \
        hires_dice_ce hires_cldice lores_dice_ce lores_cldice

all: summary

# ==============================================================
# 1. SETUP — create raw/train/<id>/ dirs with symlinks to data
#    Must run on a machine that can see the /rsrch3/... paths.
# ==============================================================
setup: datasets/hires/raw/train datasets/lores/raw/train

datasets/hires/raw/train: data/hires.csv scripts/make_dataset.py
	python3 scripts/make_dataset.py data/hires.csv datasets/hires

datasets/lores/raw/train: data/lores.csv scripts/make_dataset.py
	python3 scripts/make_dataset.py data/lores.csv datasets/lores

# ==============================================================
# 2. ANALYZE — one run per dataset; produces config.json
# ==============================================================
analyze: results/hires_base/config.json results/lores_base/config.json

results/hires_base/config.json: datasets/hires/raw/train datasets/hires/dataset.json
	mkdir -p results/hires_base
	mist_analyze \
		--data datasets/hires/dataset.json \
		--results results/hires_base \
		--nfolds $(NFOLDS) \
		--num-workers-analyze $(WORKERS)

results/lores_base/config.json: datasets/lores/raw/train datasets/lores/dataset.json
	mkdir -p results/lores_base
	mist_analyze \
		--data datasets/lores/dataset.json \
		--results results/lores_base \
		--nfolds $(NFOLDS) \
		--num-workers-analyze $(WORKERS)

# ==============================================================
# 3. PREPROCESS — one run per dataset; numpy arrays are shared
#    across loss-function experiments on the same dataset.
# ==============================================================
preprocess: numpy/hires/.preprocessed numpy/lores/.preprocessed

numpy/hires/.preprocessed: results/hires_base/config.json
	mist_preprocess \
		--results results/hires_base \
		--numpy numpy/hires \
		--num-workers-preprocess $(WORKERS)
	touch numpy/hires/.preprocessed

numpy/lores/.preprocessed: results/lores_base/config.json
	mist_preprocess \
		--results results/lores_base \
		--numpy numpy/lores \
		--num-workers-preprocess $(WORKERS)
	touch numpy/lores/.preprocessed

# ==============================================================
# 4. INIT EXPERIMENTS — copy analysis artifacts into per-experiment
#    results dirs.  mist_train overwrites config.json with the
#    CLI-specified loss, so each experiment needs its own copy.
# ==============================================================
define INIT_RULE
results/$(1)_$(2)/config.json: results/$(1)_base/config.json
	python3 scripts/init_experiment.py results/$(1)_base results/$(1)_$(2)
endef
$(foreach ds,$(DATASETS),$(foreach loss,$(LOSSES),$(eval $(call INIT_RULE,$(ds),$(loss)))))

experiments: $(foreach ds,$(DATASETS),$(foreach loss,$(LOSSES),results/$(ds)_$(loss)/config.json))

# ==============================================================
# 5. TRAIN — one job per (dataset, loss) combination
# ==============================================================
define TRAIN_RULE
results/$(1)_$(2)/results.csv: results/$(1)_$(2)/config.json numpy/$(1)/.preprocessed
	mist_train \
		--results results/$(1)_$(2) \
		--numpy numpy/$(1) \
		--model $(MODEL) \
		--loss $(2) \
		--epochs $(EPOCHS) \
		--batch-size-per-gpu $(BATCH) \
		--num-workers-evaluate $(WORKERS)
endef
$(foreach ds,$(DATASETS),$(foreach loss,$(LOSSES),$(eval $(call TRAIN_RULE,$(ds),$(loss)))))

TRAIN_TARGETS := $(foreach ds,$(DATASETS),$(foreach loss,$(LOSSES),results/$(ds)_$(loss)/results.csv))

# Named per-experiment targets — run one experiment end-to-end inside Docker.
# $$ defers DOCKER_RUN/MIST_PARAMS expansion past call so they resolve in the recipe.
define EXP_TARGET
$(1)_$(2): docker-build
	$$(DOCKER_RUN) make results/$(1)_$(2)/results.csv $$(MIST_PARAMS)
endef
$(foreach ds,$(DATASETS),$(foreach loss,$(LOSSES),$(eval $(call EXP_TARGET,$(ds),$(loss)))))

train: $(TRAIN_TARGETS)

# ==============================================================
# 6. SUMMARY — comparison table
# ==============================================================
results/comparison.md: $(TRAIN_TARGETS) scripts/collect_results.py
	python3 scripts/collect_results.py results/comparison.md $(TRAIN_TARGETS)

summary: results/comparison.md

# ==============================================================
# CLEAN
# ==============================================================
clean:
	rm -rf results/ numpy/

distclean: clean
	rm -rf datasets/hires/raw datasets/lores/raw

# ==============================================================
# DOCKER
# ==============================================================
docker-build:
	docker build -t $(IMAGE) .

docker-run: docker-build
	$(DOCKER_RUN) make all $(MIST_PARAMS)

