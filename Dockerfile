FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "mist-medical[train]" \
        pandas \
        tabulate

WORKDIR /workspace

COPY Makefile .
COPY scripts/ scripts/
COPY data/ data/
COPY datasets/ datasets/
