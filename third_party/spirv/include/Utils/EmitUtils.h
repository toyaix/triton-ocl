#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Parse array attributes.
SmallVector<int64_t, 8> getIntArrayAttrValue(Operation *op, StringRef name) {
  SmallVector<int64_t, 8> array;
  if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(name)) {
    for (auto attr : arrayAttr)
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr))
        array.push_back(intAttr.getInt());
      else
        return SmallVector<int64_t, 8>();
    return array;
  } else
    return SmallVector<int64_t, 8>();
}

static std::string getDataTypeName(Type type) {
  auto valType = type;

  // Handle aggregated types, including memref, vector, and stream.
  if (auto arrayType = mlir::dyn_cast<MemRefType>(valType))
    return getDataTypeName(arrayType.getElementType());

  // TODO: UnrankedMemRefType only use in argument
  if (auto arrayType = mlir::dyn_cast<UnrankedMemRefType>(valType))
    return "__global " + getDataTypeName(arrayType.getElementType()) + "*";

  // Refer to
  // https://registry.khronos.org/OpenCL/specs/3.0-unified/pdf/OpenCL_C.pdf
  // Handle scalar types, including float and integer.
  if (mlir::isa<Float32Type>(valType))
    return "float";
  else if (mlir::isa<Float64Type>(valType))
    return "double";
  else if (mlir::isa<IndexType>(valType))
    return "int";
  else if (auto intType = mlir::dyn_cast<IntegerType>(valType)) {
    if (intType.getWidth() == 1)
      return "bool";
    if (intType.getWidth() == 32 && !intType.isUnsigned())
      return "int";
    if (intType.getWidth() == 64 && !intType.isUnsigned())
      return "long";
    if (intType.getWidth() == 32 && intType.isUnsigned())
      return "unit";
    if (intType.getWidth() == 64 && intType.isUnsigned())
      return "ulong";
  }
  llvm_unreachable("Unsupported type.");
}

template <typename ConcreteType, typename ResultType, typename... ExtraArgs>
class HLSVisitorBase {
public:
  ResultType dispatchVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // Function operations.
            func::CallOp, func::ReturnOp,

            // SCF statements.
            scf::ForOp, scf::IfOp, scf::ParallelOp, scf::ReduceOp,
            scf::ReduceReturnOp, scf::YieldOp,

            // Affine statements.
            affine::AffineForOp, affine::AffineIfOp, affine::AffineParallelOp,
            affine::AffineApplyOp, affine::AffineMaxOp, affine::AffineMinOp,
            affine::AffineLoadOp, affine::AffineStoreOp,
            affine::AffineVectorLoadOp, affine::AffineVectorStoreOp,
            affine::AffineYieldOp,

            // Memref statements.
            memref::AllocOp, memref::AllocaOp, memref::LoadOp, memref::StoreOp,
            memref::DeallocOp, memref::CopyOp, memref::ReinterpretCastOp,
            memref::SubViewOp,

            // Unary expressions.
            math::AbsIOp, math::AbsFOp, math::CeilOp, math::CosOp, math::SinOp,
            math::TanhOp, math::SqrtOp, math::RsqrtOp, math::ExpOp,
            math::Exp2Op, math::LogOp, math::Log2Op, math::Log10Op,
            arith::NegFOp,

            // Float binary expressions.
            arith::CmpFOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
            arith::DivFOp, arith::RemFOp, arith::MaximumFOp, arith::MinimumFOp,
            math::PowFOp,

            // Integer binary expressions.
            arith::CmpIOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
            arith::DivSIOp, arith::RemSIOp, arith::DivUIOp, arith::RemUIOp,
            arith::XOrIOp, arith::AndIOp, arith::OrIOp, arith::ShLIOp,
            arith::ShRSIOp, arith::ShRUIOp, arith::MaxSIOp, arith::MinSIOp,
            arith::MaxUIOp, arith::MinUIOp,

            // Special expressions.
            arith::SelectOp, arith::ConstantOp, arith::TruncIOp,
            arith::TruncFOp, arith::ExtUIOp, arith::ExtSIOp, arith::IndexCastOp,
            arith::UIToFPOp, arith::SIToFPOp, arith::FPToSIOp, arith::FPToUIOp,

            // GPU expressions.
            gpu::GlobalIdOp>([&](auto opNode) -> ResultType {
          return thisCast->visitOp(opNode, args...);
        })
        .Default([&](auto opNode) -> ResultType {
          return thisCast->visitInvalidOp(op, args...);
        });
  }

  /// This callback is invoked on any invalid operations.
  ResultType visitInvalidOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("is unsupported operation.");
    abort();
  }

  /// This callback is invoked on any operations that are not handled by the
  /// concrete visitor.
  ResultType visitUnhandledOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitOp(OPTYPE op, ExtraArgs... args) {                           \
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);   \
  }

  // Control flow operations.
  HANDLE(func::CallOp);
  HANDLE(func::ReturnOp);

  // SCF statements.
  HANDLE(scf::ForOp);
  HANDLE(scf::IfOp);
  HANDLE(scf::ParallelOp);
  HANDLE(scf::ReduceOp);
  HANDLE(scf::ReduceReturnOp);
  HANDLE(scf::YieldOp);

  // Affine statements.
  HANDLE(affine::AffineForOp);
  HANDLE(affine::AffineIfOp);
  HANDLE(affine::AffineParallelOp);
  HANDLE(affine::AffineApplyOp);
  HANDLE(affine::AffineMaxOp);
  HANDLE(affine::AffineMinOp);
  HANDLE(affine::AffineLoadOp);
  HANDLE(affine::AffineStoreOp);
  HANDLE(affine::AffineVectorLoadOp);
  HANDLE(affine::AffineVectorStoreOp);
  HANDLE(affine::AffineYieldOp);

  // Memref statements.
  HANDLE(memref::AllocOp);
  HANDLE(memref::AllocaOp);
  HANDLE(memref::LoadOp);
  HANDLE(memref::StoreOp);
  HANDLE(memref::DeallocOp);
  HANDLE(memref::CopyOp);
  HANDLE(memref::ReinterpretCastOp);
  HANDLE(memref::SubViewOp);

  // Unary expressions.
  HANDLE(math::AbsIOp);
  HANDLE(math::AbsFOp);
  HANDLE(math::CeilOp);
  HANDLE(math::CosOp);
  HANDLE(math::SinOp);
  HANDLE(math::TanhOp);
  HANDLE(math::SqrtOp);
  HANDLE(math::RsqrtOp);
  HANDLE(math::ExpOp);
  HANDLE(math::Exp2Op);
  HANDLE(math::LogOp);
  HANDLE(math::Log2Op);
  HANDLE(math::Log10Op);
  HANDLE(arith::NegFOp);

  // Float binary expressions.
  HANDLE(arith::CmpFOp);
  HANDLE(arith::AddFOp);
  HANDLE(arith::SubFOp);
  HANDLE(arith::MulFOp);
  HANDLE(arith::DivFOp);
  HANDLE(arith::RemFOp);
  HANDLE(arith::MaximumFOp);
  HANDLE(arith::MinimumFOp);
  HANDLE(math::PowFOp);

  // Integer binary expressions.
  HANDLE(arith::CmpIOp);
  HANDLE(arith::AddIOp);
  HANDLE(arith::SubIOp);
  HANDLE(arith::MulIOp);
  HANDLE(arith::DivSIOp);
  HANDLE(arith::RemSIOp);
  HANDLE(arith::DivUIOp);
  HANDLE(arith::RemUIOp);
  HANDLE(arith::XOrIOp);
  HANDLE(arith::AndIOp);
  HANDLE(arith::OrIOp);
  HANDLE(arith::ShLIOp);
  HANDLE(arith::ShRSIOp);
  HANDLE(arith::ShRUIOp);
  HANDLE(arith::MaxSIOp);
  HANDLE(arith::MinSIOp);
  HANDLE(arith::MaxUIOp);
  HANDLE(arith::MinUIOp);

  // Special expressions.
  HANDLE(arith::SelectOp);
  HANDLE(arith::ConstantOp);
  HANDLE(arith::TruncIOp);
  HANDLE(arith::TruncFOp);
  HANDLE(arith::ExtUIOp);
  HANDLE(arith::ExtSIOp);
  HANDLE(arith::ExtFOp);
  HANDLE(arith::IndexCastOp);
  HANDLE(arith::UIToFPOp);
  HANDLE(arith::SIToFPOp);
  HANDLE(arith::FPToUIOp);
  HANDLE(arith::FPToSIOp);

  // GPU expressions.
  HANDLE(gpu::GlobalIdOp);
#undef HANDLE
};

//===----------------------------------------------------------------------===//
// Some Base Classes
//===----------------------------------------------------------------------===//

namespace {
/// This class maintains the mutable state that cross-cuts and is shared by the
/// various emitters.
class ScaleHLSEmitterState {
public:
  explicit ScaleHLSEmitterState(raw_ostream &os) : os(os) {}

  // The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

  // This table contains all declared values.
  DenseMap<Value, SmallString<8>> nameTable;

private:
  ScaleHLSEmitterState(const ScaleHLSEmitterState &) = delete;
  void operator=(const ScaleHLSEmitterState &) = delete;
};
} // namespace

namespace {
/// This is the base class for all of the HLSCpp Emitter components.
class ScaleHLSEmitterBase {
public:
  explicit ScaleHLSEmitterBase(ScaleHLSEmitterState &state)
      : state(state), os(state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  raw_ostream &indent() { return os.indent(state.currentIndent); }

  void addIndent() { state.currentIndent += 2; }
  void reduceIndent() { state.currentIndent -= 2; }

  // All of the mutable state we are maintaining.
  ScaleHLSEmitterState &state;

  // The stream to emit to.
  raw_ostream &os;

  /// Value name management methods.
  SmallString<8> addName(Value val, bool isPtr = false);

  SmallString<8> addAlias(Value val, Value alias);

  SmallString<8> getName(Value val);

  bool isDeclared(Value val) {
    if (getName(val).empty()) {
      return false;
    }
    return true;
  }

private:
  ScaleHLSEmitterBase(const ScaleHLSEmitterBase &) = delete;
  void operator=(const ScaleHLSEmitterBase &) = delete;
};
} // namespace

// TODO: update naming rule.
SmallString<8> ScaleHLSEmitterBase::addName(Value val, bool isPtr) {
  assert(!isDeclared(val) && "has been declared before.");

  SmallString<8> valName;
  if (isPtr)
    valName += "*";

  valName += StringRef("var_" + std::to_string(state.nameTable.size()));
  state.nameTable[val] = valName;

  return valName;
}

SmallString<8> ScaleHLSEmitterBase::addAlias(Value val, Value alias) {
  assert(!isDeclared(alias) && "has been declared before.");
  assert(isDeclared(val) && "hasn't been declared before.");

  auto valName = getName(val);
  state.nameTable[alias] = valName;

  return valName;
}

static SmallString<8> getConstantString(Type type, Attribute attr) {
  SmallString<8> string;
  if (type.isInteger(1)) {
    auto value = mlir::cast<BoolAttr>(attr).getValue();
    string.append(value ? "true" : "false");
  } else if (type.isIndex()) {
    auto value = mlir::cast<IntegerAttr>(attr).getInt();
    string.append(std::to_string(value));

  } else if (auto floatType = mlir::dyn_cast<FloatType>(type)) {
    if (floatType.getWidth() == 32) {
      auto value = mlir::cast<FloatAttr>(attr).getValue().convertToFloat();
      string.append(std::isfinite(value)
                        ? std::to_string(value)
                        : (value > 0 ? "INFINITY" : "-INFINITY"));
    } else if (floatType.getWidth() == 64) {
      auto value = mlir::cast<FloatAttr>(attr).getValue().convertToDouble();
      string.append(std::isfinite(value)
                        ? std::to_string(value)
                        : (value > 0 ? "INFINITY" : "-INFINITY"));
    }
  } else if (auto intType = mlir::dyn_cast<IntegerType>(type)) {
    if (intType.isSigned()) {
      auto value = mlir::cast<IntegerAttr>(attr).getValue().getSExtValue();
      string.append(std::to_string(value));
    } else if (intType.isUnsigned()) {
      auto value = mlir::cast<IntegerAttr>(attr).getValue().getZExtValue();
      string.append(std::to_string(value));
    } else {
      auto value = mlir::cast<IntegerAttr>(attr).getInt();
      string.append(std::to_string(value));
    }
  }
  return string;
}

SmallString<8> ScaleHLSEmitterBase::getName(Value val) {
  // For constant scalar operations, the constant number will be returned rather
  // than the value name.
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>())
    if (!mlir::isa<ShapedType>(constOp.getType())) {
      auto string = getConstantString(constOp.getType(), constOp.getValue());
      if (string.empty())
        constOp.emitOpError("constant has invalid value");
      return string;
    }
  return state.nameTable.lookup(val);
}

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Declaration
//===----------------------------------------------------------------------===//

namespace {
class ModuleEmitter : public ScaleHLSEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(ScaleHLSEmitterState &state)
      : ScaleHLSEmitterBase(state) {}

  template <typename AssignOpType> void emitAssign(AssignOpType op);

  /// Control flow operation emitters.
  void emitCall(func::CallOp op);

  /// SCF statement emitters.
  void emitScfFor(scf::ForOp op);
  void emitScfIf(scf::IfOp op);
  void emitScfYield(scf::YieldOp op);

  /// Affine statement emitters.
  void emitAffineFor(affine::AffineForOp op);
  void emitAffineIf(affine::AffineIfOp op);
  void emitAffineParallel(affine::AffineParallelOp op);
  void emitAffineApply(affine::AffineApplyOp op);
  template <typename OpType>
  void emitAffineMaxMin(OpType op, const char *syntax);
  void emitAffineLoad(affine::AffineLoadOp op);
  void emitAffineStore(affine::AffineStoreOp op);
  void emitAffineYield(affine::AffineYieldOp op);

  /// Memref-related statement emitters.
  template <typename OpType> void emitAlloc(OpType op);
  void emitLoad(memref::LoadOp op);
  void emitStore(memref::StoreOp op);
  void emitMemCpyValue(Value val);
  void emitMemCpy(memref::CopyOp op);
  template <typename OpType> void emitReshape(OpType op);

  /// Standard expression emitters.
  void emitUnary(Operation *op, const char *syntax);
  void emitBinary(Operation *op, const char *syntax);
  template <typename OpType> void emitMaxMin(OpType op, const char *syntax);

  /// Special expression emitters.
  void emitSelect(arith::SelectOp op);
  template <typename OpType> void emitConstant(OpType op);

  // Gpu op emitters.
  void emitGlobalId(gpu::GlobalIdOp op);

  void emitOneAssign(arith::IndexCastOp op);

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);

private:
  /// Helper to get the string indices of TransferRead/Write operations.
  template <typename TransferOpType>
  SmallVector<SmallString<8>, 4> getTransferIndices(TransferOpType op);

  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 bool isRef = false);
  void emitArrayDecl(Value array);
  unsigned emitNestedLoopHeader(Value val);
  void emitNestedLoopFooter(unsigned rank);
  void emitInfoAndNewLine(Operation *op);

  /// MLIR component and HLS C++ pragma emitters.
  void emitBlock(Block &block);
  void emitFunction(func::FuncOp func);

  unsigned numDSPs = 0;
};
} // namespace

//===----------------------------------------------------------------------===//
// AffineEmitter Class
//===----------------------------------------------------------------------===//

namespace {
class AffineExprEmitter : public ScaleHLSEmitterBase,
                          public AffineExprVisitor<AffineExprEmitter> {
public:
  using operand_range = Operation::operand_range;
  explicit AffineExprEmitter(ScaleHLSEmitterState &state, unsigned numDim,
                             operand_range operands)
      : ScaleHLSEmitterBase(state), numDim(numDim), operands(operands) {}

  void visitAddExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "+"); }
  void visitMulExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "*"); }
  void visitModExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "%"); }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    emitAffineBinary(expr, "/");
  }
  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    // This is super inefficient.
    os << "(";
    visit(expr.getLHS());
    os << " + ";
    visit(expr.getRHS());
    os << " - 1) / ";
    visit(expr.getRHS());
    os << ")";
  }

  void visitConstantExpr(AffineConstantExpr expr) { os << expr.getValue(); }

  void visitDimExpr(AffineDimExpr expr) {
    os << getName(operands[expr.getPosition()]);
  }
  void visitSymbolExpr(AffineSymbolExpr expr) {
    os << getName(operands[numDim + expr.getPosition()]);
  }

  /// Affine expression emitters.
  void emitAffineBinary(AffineBinaryOpExpr expr, const char *syntax) {
    os << "(";
    if (auto constRHS = mlir::dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if ((unsigned)*syntax == (unsigned)*"*" && constRHS.getValue() == -1) {
        os << "-";
        visit(expr.getLHS());
        os << ")";
        return;
      }
      if ((unsigned)*syntax == (unsigned)*"+" && constRHS.getValue() < 0) {
        visit(expr.getLHS());
        os << " - ";
        os << -constRHS.getValue();
        os << ")";
        return;
      }
    }
    if (auto binaryRHS = mlir::dyn_cast<AffineBinaryOpExpr>(expr.getRHS())) {
      if (auto constRHS =
              mlir::dyn_cast<AffineConstantExpr>(binaryRHS.getRHS())) {
        if ((unsigned)*syntax == (unsigned)*"+" && constRHS.getValue() == -1 &&
            binaryRHS.getKind() == AffineExprKind::Mul) {
          visit(expr.getLHS());
          os << " - ";
          visit(binaryRHS.getLHS());
          os << ")";
          return;
        }
      }
    }
    visit(expr.getLHS());
    os << " " << syntax << " ";
    visit(expr.getRHS());
    os << ")";
  }

  void emitAffineExpr(AffineExpr expr) { visit(expr); }

private:
  unsigned numDim;
  operand_range operands;
};
} // namespace

//===----------------------------------------------------------------------===//
// StmtVisitor, ExprVisitor, and PragmaVisitor Classes
//===----------------------------------------------------------------------===//

namespace {
class StmtVisitor : public HLSVisitorBase<StmtVisitor, bool> {
public:
  StmtVisitor(ModuleEmitter &emitter) : emitter(emitter) {}
  using HLSVisitorBase::visitOp;

  /// Function operations.
  bool visitOp(func::CallOp op) { return emitter.emitCall(op), true; }
  bool visitOp(func::ReturnOp op) { return true; }

  /// SCF statements.
  bool visitOp(scf::ForOp op) { return emitter.emitScfFor(op), true; };
  bool visitOp(scf::IfOp op) { return emitter.emitScfIf(op), true; };
  bool visitOp(scf::ParallelOp op) { return false; };
  bool visitOp(scf::ReduceOp op) { return false; };
  bool visitOp(scf::ReduceReturnOp op) { return false; };
  bool visitOp(scf::YieldOp op) { return emitter.emitScfYield(op), true; };

  /// Affine statements.
  bool visitOp(affine::AffineForOp op) {
    return emitter.emitAffineFor(op), true;
  }
  bool visitOp(affine::AffineIfOp op) { return emitter.emitAffineIf(op), true; }
  bool visitOp(affine::AffineParallelOp op) {
    return emitter.emitAffineParallel(op), true;
  }
  bool visitOp(affine::AffineApplyOp op) {
    return emitter.emitAffineApply(op), true;
  }
  bool visitOp(affine::AffineMaxOp op) {
    return emitter.emitAffineMaxMin(op, "max"), true;
  }
  bool visitOp(affine::AffineMinOp op) {
    return emitter.emitAffineMaxMin(op, "min"), true;
  }
  bool visitOp(affine::AffineLoadOp op) {
    return emitter.emitAffineLoad(op), true;
  }
  bool visitOp(affine::AffineStoreOp op) {
    return emitter.emitAffineStore(op), true;
  }
  bool visitOp(affine::AffineVectorLoadOp op) { return false; }
  bool visitOp(affine::AffineVectorStoreOp op) { return false; }
  bool visitOp(affine::AffineYieldOp op) {
    return emitter.emitAffineYield(op), true;
  }

  /// Memref statements.
  bool visitOp(memref::AllocOp op) { return emitter.emitAlloc(op), true; }
  bool visitOp(memref::AllocaOp op) { return emitter.emitAlloc(op), true; }
  bool visitOp(memref::LoadOp op) { return emitter.emitLoad(op), true; }
  bool visitOp(memref::StoreOp op) { return emitter.emitStore(op), true; }
  bool visitOp(memref::DeallocOp op) { return true; }
  bool visitOp(memref::CopyOp op) { return emitter.emitMemCpy(op), true; }
  bool visitOp(memref::ReinterpretCastOp op) { return true; }
  bool visitOp(memref::SubViewOp op) { return true; }

  bool visitOp(gpu::GlobalIdOp op) { return emitter.emitGlobalId(op), true; }

private:
  ModuleEmitter &emitter;
};
} // namespace

namespace {
class ExprVisitor : public HLSVisitorBase<ExprVisitor, bool> {
public:
  ExprVisitor(ModuleEmitter &emitter) : emitter(emitter) {}
  using HLSVisitorBase::visitOp;

  /// Unary expressions.
  bool visitOp(math::AbsIOp op) { return emitter.emitUnary(op, "abs"), true; }
  bool visitOp(math::AbsFOp op) { return emitter.emitUnary(op, "abs"), true; }
  bool visitOp(math::CeilOp op) { return emitter.emitUnary(op, "ceil"), true; }
  bool visitOp(math::CosOp op) { return emitter.emitUnary(op, "cos"), true; }
  bool visitOp(math::SinOp op) { return emitter.emitUnary(op, "sin"), true; }
  bool visitOp(math::TanhOp op) { return emitter.emitUnary(op, "tanh"), true; }
  bool visitOp(math::SqrtOp op) { return emitter.emitUnary(op, "sqrt"), true; }
  bool visitOp(math::RsqrtOp op) {
    return emitter.emitUnary(op, "1.0 / sqrt"), true;
  }
  bool visitOp(math::ExpOp op) { return emitter.emitUnary(op, "exp"), true; }
  bool visitOp(math::Exp2Op op) { return emitter.emitUnary(op, "exp2"), true; }
  bool visitOp(math::LogOp op) { return emitter.emitUnary(op, "log"), true; }
  bool visitOp(math::Log2Op op) { return emitter.emitUnary(op, "log2"), true; }
  bool visitOp(math::Log10Op op) {
    return emitter.emitUnary(op, "log10"), true;
  }
  bool visitOp(arith::NegFOp op) { return emitter.emitUnary(op, "-"), true; }

  /// Float binary expressions.
  bool visitOp(arith::CmpFOp op);
  bool visitOp(arith::AddFOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(arith::SubFOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(arith::MulFOp op) { return emitter.emitBinary(op, "*"), true; }
  bool visitOp(arith::DivFOp op) { return emitter.emitBinary(op, "/"), true; }
  bool visitOp(arith::RemFOp op) { return emitter.emitBinary(op, "%"), true; }
  bool visitOp(arith::MaximumFOp op) {
    return emitter.emitMaxMin(op, "max"), true;
  }
  bool visitOp(arith::MinimumFOp op) {
    return emitter.emitMaxMin(op, "min"), true;
  }
  bool visitOp(math::PowFOp op) { return emitter.emitMaxMin(op, "pow"), true; }

  /// Integer binary expressions.
  bool visitOp(arith::CmpIOp op);
  bool visitOp(arith::AddIOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(arith::SubIOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(arith::MulIOp op) { return emitter.emitBinary(op, "*"), true; }
  bool visitOp(arith::DivSIOp op) { return emitter.emitBinary(op, "/"), true; }
  bool visitOp(arith::RemSIOp op) { return emitter.emitBinary(op, "%"), true; }
  bool visitOp(arith::DivUIOp op) { return emitter.emitBinary(op, "/"), true; }
  bool visitOp(arith::RemUIOp op) { return emitter.emitBinary(op, "%"), true; }
  bool visitOp(arith::XOrIOp op) { return emitter.emitBinary(op, "^"), true; }
  bool visitOp(arith::AndIOp op) { return emitter.emitBinary(op, "&"), true; }
  bool visitOp(arith::OrIOp op) { return emitter.emitBinary(op, "|"), true; }
  bool visitOp(arith::ShLIOp op) { return emitter.emitBinary(op, "<<"), true; }
  bool visitOp(arith::ShRSIOp op) { return emitter.emitBinary(op, ">>"), true; }
  bool visitOp(arith::ShRUIOp op) { return emitter.emitBinary(op, ">>"), true; }
  bool visitOp(arith::MaxSIOp op) {
    return emitter.emitMaxMin(op, "max"), true;
  }
  bool visitOp(arith::MinSIOp op) {
    return emitter.emitMaxMin(op, "min"), true;
  }
  bool visitOp(arith::MaxUIOp op) {
    return emitter.emitMaxMin(op, "max"), true;
  }
  bool visitOp(arith::MinUIOp op) {
    return emitter.emitMaxMin(op, "min"), true;
  }

  /// Special expressions.
  bool visitOp(arith::SelectOp op) { return emitter.emitSelect(op), true; }
  bool visitOp(arith::ConstantOp op) { return emitter.emitConstant(op), true; }
  bool visitOp(arith::IndexCastOp op) { return emitter.emitOneAssign(op), true; }
  bool visitOp(arith::UIToFPOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::SIToFPOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::FPToUIOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::FPToSIOp op) { return emitter.emitAssign(op), true; }

  /// TODO: Figure out whether these ops need to be separately handled.
  bool visitOp(arith::TruncIOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::TruncFOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::ExtUIOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::ExtSIOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::ExtFOp op) { return emitter.emitAssign(op), true; }

private:
  ModuleEmitter &emitter;
};
} // namespace

bool ExprVisitor::visitOp(arith::CmpFOp op) {
  switch (op.getPredicate()) {
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    return emitter.emitBinary(op, "=="), true;
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    return emitter.emitBinary(op, "!="), true;
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    return emitter.emitBinary(op, "<"), true;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    return emitter.emitBinary(op, "<="), true;
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    return emitter.emitBinary(op, ">"), true;
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    return emitter.emitBinary(op, ">="), true;
  default:
    op.emitError("has unsupported compare type.");
    return false;
  }
}

bool ExprVisitor::visitOp(arith::CmpIOp op) {
  switch (op.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return emitter.emitBinary(op, "=="), true;
  case arith::CmpIPredicate::ne:
    return emitter.emitBinary(op, "!="), true;
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return emitter.emitBinary(op, "<"), true;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return emitter.emitBinary(op, "<="), true;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return emitter.emitBinary(op, ">"), true;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return emitter.emitBinary(op, ">="), true;
  }
  return false;
}

/// C++ component emitters.
void ModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                              bool isRef) {
  assert(!(rank && isPtr) && "should be either an array or a pointer.");

  // Value has been declared before or is a constant number.
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }

  // Emit the type of the value.
  os << getDataTypeName(val.getType()) << " ";
  if (isRef)
    os << "&";

  // Add the new value to nameTable and emit its name.
  os << addName(val, isPtr);
  for (unsigned i = 0; i < rank; ++i)
    os << "[iv" << i << "]";
}

void ModuleEmitter::emitArrayDecl(Value array) {
  assert(!isDeclared(array) && "has been declared before.");
  auto arrayType = mlir::dyn_cast<MemRefType>(array.getType());

  if (arrayType.hasStaticShape()) {
    emitValue(array);
    for (auto &shape : arrayType.getShape()) {
      os << "[" << shape << "]";
    }
  } else
    emitValue(array, /*rank=*/0, /*isPtr=*/true);
}

unsigned ModuleEmitter::emitNestedLoopHeader(Value val) {
  unsigned rank = 0;

  if (auto type = mlir::dyn_cast<MemRefType>(val.getType())) {
    if (!type.hasStaticShape()) {
      emitError(val.getDefiningOp(), "is unranked or has dynamic shape.");
      return 0;
    }

    // Declare a new array.
    if (!isDeclared(val)) {
      indent();
      emitArrayDecl(val);
      os << ";\n";
      // TODO: More precise control here. Now we assume vectors are always
      // completely partitioned at all dimensions.
      if (mlir::isa<VectorType>(type)) {
        indent() << "#pragma HLS array_partition variable=";
        emitValue(val);
        os << " complete dim=0\n";
      }
    }

    // Create nested loop.
    unsigned dimIdx = 0;
    for (auto &shape : type.getShape()) {
      indent() << "for (int iv" << dimIdx << " = 0; ";
      os << "iv" << dimIdx << " < " << shape << "; ";
      os << "++iv" << dimIdx++ << ") {\n";

      addIndent();
      // TODO: More precise control here. Now we assume vectorization loops are
      // always fully unrolled.
      if (mlir::isa<VectorType>(type))
        indent() << "#pragma HLS unroll\n";
    }
    rank = type.getRank();
  }

  return rank;
}

void ModuleEmitter::emitNestedLoopFooter(unsigned rank) {
  for (unsigned i = 0; i < rank; ++i) {
    reduceIndent();

    indent() << "}\n";
  }
}

void ModuleEmitter::emitInfoAndNewLine(Operation *op) {
  os << "\t//";
  // Print line number.
  if (auto loc = mlir::dyn_cast<FileLineColLoc>(op->getLoc()))
    os << " L" << loc.getLine();
  os << "\n";
}

/// MLIR component and HLS C++ pragma emitters.
void ModuleEmitter::emitBlock(Block &block) {
  for (auto &op : block) {
    if (ExprVisitor(*this).dispatchVisitor(&op))
      continue;

    if (StmtVisitor(*this).dispatchVisitor(&op))
      continue;

    emitError(&op, "can't be correctly emitted.");
  }
}
