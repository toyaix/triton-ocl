#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "spirv/include/Conversion/TritonToLinalg/Passes.h"
#include "spirv/include/Conversion/LinalgToAffineLoops/Passes.h"
#include "spirv/include/Conversion/AffineToLLVMSPV/Passes.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllPasses.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::triton::spirv::registerTritonToLinalgPass();
  mlir::triton::spirv::registerLinalgToAffineLoops();
  mlir::triton::spirv::registerAffineToLLVMSPV();

  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
                  mlir::math::MathDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::linalg::LinalgDialect,
                  mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::memref::MemRefDialect, ::mlir::gpu::GPUDialect,
                  mlir::LLVM::LLVMDialect, mlir::affine::AffineDialect>();

  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "spirv-opt test driver\n", registry));
}
