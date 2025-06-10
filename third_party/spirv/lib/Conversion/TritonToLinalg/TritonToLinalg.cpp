#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "spirv/include/Analysis/UseAnalysis.h"
#include "spirv/include/Conversion/TritonToLinalg/ConversionPatterns.hpp"
#include "spirv/include/Conversion/TritonToLinalg/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-linalg"

namespace mlir::triton::spirv {
#define GEN_PASS_DEF_TRITONTOLINALG
#include "spirv/include/Conversion/TritonToLinalg/Passes.h.inc"
} // namespace mlir::triton::spirv

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::spirv;

namespace {

void populateTritonToLinalgConversionPatterns(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
  populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
      patterns, typeConverter);
  patterns.add<MetaOpConverter>(patterns.getContext());
  patterns.add<StoreConverter>(patterns.getContext());
  patterns.add<LegacyAddPtrConverter>(patterns.getContext());
  patterns.add<GetProgramIDConverter>(patterns.getContext());
  patterns.add<YieldConverter>(patterns.getContext());
  patterns.add<LoadConverter>(patterns.getContext());
  patterns.add<LoopConverter>(patterns.getContext());
  patterns.add<BroadcastConverter>(patterns.getContext());
  patterns.add<SplatConverter>(patterns.getContext());
  patterns.add<DenseConstantConverter>(patterns.getContext());
  linalg::populateElementwiseToLinalgConversionPatterns(patterns);
}

class TritonTypeConverter : public TypeConverter {
public:
  TritonTypeConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    addConversion([](TensorType tensorType) -> Type {
      auto elemType = tensorType.getElementType();
      if (auto ptrType = dyn_cast<triton::PointerType>(elemType)) {
        elemType = ptrType.getPointeeType();
      }
      return MemRefType::get(tensorType.getShape(), elemType);
    });
  }
};

struct TritonToLinalg
    : public mlir::triton::spirv::impl::TritonToLinalgBase<TritonToLinalg> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, ::mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    moduleOp.walk([this](triton::FuncOp op) {
      if (failed(runUseAnalysis(op))) {
        signalPassFailure();
      }
    });

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonTypeConverter tritonTypeConverter;

    target.addLegalDialect<func::FuncDialect, arith::ArithDialect,
                           math::MathDialect, linalg::LinalgDialect,
                           affine::AffineDialect, scf::SCFDialect,
                           cf::ControlFlowDialect, tensor::TensorDialect,
                           bufferization::BufferizationDialect,
                           memref::MemRefDialect, ::mlir::gpu::GPUDialect>();

    target.addIllegalOp<triton::GetProgramIdOp>();

    target.addLegalOp<ModuleOp>();

    // Update function signature to use memrefs
    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
      return tritonTypeConverter.isSignatureLegal(op.getFunctionType());
    });

    // Lower dense constant to linalg.fill
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      if (!isa<RankedTensorType>(op.getResult().getType())) {
        return true;
      }

      if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (denseAttr.isSplat() &&
            isa<FloatType, IntegerType>(denseAttr.getElementType())) {
          return false;
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ForOp, scf::YieldOp>([](Operation *op) {
      return llvm::all_of(op->getOperandTypes(), [](Type t) {
        if (isa<triton::PointerType>(t)) {
          return false;
        }
        if (auto shapedType = dyn_cast<ShapedType>(t)) {
          return shapedType.getElementType().isIntOrFloat();
        }
        assert(t.isIntOrIndexOrFloat());
        return true;
      });
    });

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          if (op->hasAttr("MetaUse")) {
            return false;
          }

          if (isa<arith::ConstantOp>(op)) {
            return true;
          }

          bool operateOnTensors =
              llvm::all_of(op->getOperandTypes(), [](Type type) {
                return isa<RankedTensorType>(type);
              });

          return !operateOnTensors;
        });

    populateTritonToLinalgConversionPatterns(tritonTypeConverter, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();

    // Convert tt.func and tt.return into func's counterparts
    moduleOp.walk([&](triton::FuncOp func) {
      OpBuilder builder(func);

      auto name = func.getName();
      auto type = func.getFunctionType();

      SmallVector<DictionaryAttr> argAttrs, resAttrs;
      func.getAllArgAttrs(argAttrs);
      func.getAllResultAttrs(resAttrs);

      auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, type);
      funcFunc.setAllArgAttrs(argAttrs);
      funcFunc.setAllResultAttrs(resAttrs);

      auto &funcFuncBody = funcFunc.getBody();
      auto &funcBody = func.getBody();

      IRMapping map;
      funcBody.cloneInto(&funcFuncBody, map);

      for (Block &block : funcFuncBody.getBlocks()) {
        auto term = block.getTerminator();
        builder.setInsertionPoint(term);
        builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
        term->erase();
      }
      func.erase();
    });

    // Erase dead code and fold constants created during lowering
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton::spirv {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalg>();
}

} // namespace mlir::triton::spirv
