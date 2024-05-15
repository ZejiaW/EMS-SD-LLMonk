# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
from mpi4py import MPI

def get_world_size():
    size = 1
    if dist.is_initialized():
        size = dist.get_world_size()
    elif MPI.Is_initialized():
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
    return size


def get_rank():
    if dist.is_initialized():
        rank = dist.get_rank()
    elif MPI.Is_initialized():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    return rank


def get_device_count():
    return torch.cuda.device_count()


def get_device():
    return torch.cuda.current_device()


class ModelParallelGroup:

    def __init__(self, tensor_para_size: int, pipeline_para_size: int):
        self.check_parallel_size_validity(tensor_para_size, pipeline_para_size)

        rank = get_rank()
        device = rank % get_device_count()
        torch.cuda.set_device(device)

        # tp: tensor parallel, pp: pipeline parallel.
        self.tp_size = tensor_para_size
        self.tp_rank = rank % self.tp_size
        self.pp_size = pipeline_para_size
        self.pp_rank = rank // self.tp_size

    @staticmethod
    def check_parallel_size_validity(tensor_para_size, pipeline_para_size):
        world_size = get_world_size()
        if world_size != tensor_para_size * pipeline_para_size:
            raise ValueError(
                f'[ERROR] Invalid tensor/pipeline parallel configuration. '
                f'world_size({world_size}) != tensor_para_size({tensor_para_size})'
                f' * pipeline_para_size({pipeline_para_size})')

    @property
    def is_pipeline_first(self):
        return self.pp_rank == 0

    @property
    def is_pipeline_last(self):
        return self.pp_rank == self.pp_size - 1


_model_para_group = None


def is_model_parallel_initailized():
    return _model_para_group is not None


def initialize_model_parallel(tensor_para_size: int,
                              pipeline_para_size: int,
                              backend=dist.Backend.MPI):
    if tensor_para_size == 1 and pipeline_para_size == 1:
        return

    assert torch.cuda.is_available()
    assert not is_model_parallel_initailized(), \
        f'parallel group has been already initialized.'

    print('Initializing tensor and pipeline parallel...')
    # dist.init_process_group(backend=backend)

    global _model_para_group
    _model_para_group = ModelParallelGroup(tensor_para_size, pipeline_para_size)


def get_tensor_para_rank():
    if _model_para_group is None:
        return 0
    return _model_para_group.tp_rank


def get_tensor_para_size():
    if _model_para_group is None:
        return 1
    return _model_para_group.tp_size


def get_pipeline_para_rank():
    if _model_para_group is None:
        return 0
    return _model_para_group.pp_rank


def get_pipeline_para_size():
    if _model_para_group is None:
        return 1
    return _model_para_group.pp_size


def is_pipeline_group_first():
    return _model_para_group is None or _model_para_group.is_pipeline_first


def is_pipeline_group_last():
    return _model_para_group is None or _model_para_group.is_pipeline_last


def destroy():
    dist.destroy_process_group()
