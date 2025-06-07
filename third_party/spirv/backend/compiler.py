from triton.backends.compiler import BaseBackend, GPUTarget
import functools
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from triton._C.libtriton import spirv, ir, llvm, passes

from triton.runtime.build import _build


@dataclass(frozen=True)
class SPIRVOptions:
    backend_name: str = "spirv"
    num_warps: int = 0
    num_stages: int = 0
    num_ctas: int = 0
    num_threads: int = 0
    cluster_dims: tuple = (1, 1, 1)
    extern_libs: dict = None
    debug: bool = False
    launch_cooperative_grid: bool = False
    max_num_imprecise_acc_default: int = 0
    sanitize_overflow: bool = False

    def __post_init__(self):
        pass

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class SPIRVBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "spirv"

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        self.binary_ext = "so"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in SPIRVOptions.__dataclass_fields__.keys() if k in opts}
        return SPIRVOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self, options):
        pass

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        spirv.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_lair(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        spirv.passes.lair.triton_to_linalg(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_memir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        spirv.passes.memir.one_shot_bufferize(pm)
        spirv.passes.memir.linalg_to_affine_loops(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_llvmspvir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        spirv.passes.llvmspvir.affine_to_llvmspv(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def emit_opencl(src, metadata, opt):
        import triton._C as tc
        spirv_translate = os.path.join(tc.__path__[0], 'triton-spirv-translate')
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.memir') as fsrc:
            fsrc.write(str(src))
            fsrc.flush()
            opencl_file = fsrc.name + '.cl'
            emit_opencl_cmd = [
                spirv_translate,
                fsrc.name,
                '-triton-spirv-emit-opencl',
                '-o',
                opencl_file
            ]
            print(" ".join(emit_opencl_cmd))
            subprocess.run(emit_opencl_cmd, check=True, close_fds=False)
            with open(opencl_file, 'rb') as f:
                opencl_src = f.read()
            if os.path.exists(opencl_file):
                os.remove(opencl_file)
        return opencl_src

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["lair"] = lambda src, metadata: self.make_lair(src, metadata, options)
        stages["memir"] = lambda src, metadata: self.make_memir(src, metadata, options)
        stages["cl"] = lambda src, metadata: self.emit_opencl(src, metadata, options)


    @functools.lru_cache()
    def hash(self):
        import platform
        return f"{platform.machine()}"
