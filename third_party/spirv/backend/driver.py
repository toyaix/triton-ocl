import os
import hashlib
import importlib
import importlib.resources
import tempfile
import time

import triton
import triton._C
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

from pathlib import Path


# ------------------------
# Utils
# ------------------------


class SPIRVUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SPIRVUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        pass

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


# ------------------------
# Launcher
# ------------------------

def make_launcher(constants, signature, ids):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.

    # generate glue code
    src = f""""""
    return src


class SPIRVLauncher(object):

    def __init__(self, src, metadata):
        pass

    def __call__(self, gridX, gridY, gridZ, kernel_name, kernel_src, bound_args):
        from triton.backends.spirv.cl_utils import launch
        launch(gridX, gridY, gridZ, kernel_name, kernel_src, bound_args)


class SPIRVDriver(DriverBase):

    def __init__(self):
        self.utils = SPIRVUtils()
        self.launcher_cls = SPIRVLauncher
        super().__init__()

    def get_current_device(self):
        return 0

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu", self.get_current_device())

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        # Capability and warp size are zeros for SPIRV.
        return GPUTarget("spirv", "test", 0)

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        pass

    def clear_cache(self, cache):
        cache.zero_()
