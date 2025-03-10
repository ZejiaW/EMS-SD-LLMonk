#!/bin/bash

pip install -r ./examples/pytorch/gpt/requirement.txt

python -m pip install mpi4py

export OMPI_ALLOW_RUN_AS_ROOT=1

export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1