#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "spirv/include/Conversion/AffineToLLVMSPV/Passes.h"
#include "spirv/include/Conversion/LinalgToAffineLoops/Passes.h"
#include "spirv/include/Conversion/TritonToLinalg/Passes.h"

namespace py = pybind11;

void init_triton_spirv_passes_lair(py::module &&m) {
  m.def("triton_to_linalg", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spirv::createTritonToLinalgPass());
  });
}

void init_triton_spirv_passes_memir(py::module &&m) {
  m.def("one_shot_bufferize", [](mlir::PassManager &pm) {
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());
  });
  m.def("linalg_to_affine_loops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spirv::createLinalgToAffineLoopsPass());
  });
}

void init_triton_spirv_passes_llvmspvir(py::module &&m) {
  m.def("affine_to_llvmspv", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::spirv::createAffineToLLVMSPVPass());
  });
}

void init_triton_spirv(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_spirv_passes_lair(passes.def_submodule("lair"));
  init_triton_spirv_passes_llvmspvir(passes.def_submodule("llvmspvir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_spirv_passes_memir(passes.def_submodule("memir"));
}
