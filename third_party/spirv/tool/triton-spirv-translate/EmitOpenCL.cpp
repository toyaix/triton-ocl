//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
// Refer to
// https://github.com/UIUC-ChenLab/ScaleHLS-HIDA/blob/main/lib/Translation/EmitHLSCpp.cpp
//
//===----------------------------------------------------------------------===//

#include "spirv/include/Utils/EmitUtils.h"

using namespace mlir;

void ModuleEmitter::emitOneAssign(arith::IndexCastOp op) {
  addAlias(op.getOperand(), op.getResult());
}

template <typename AssignOpType>
void ModuleEmitter::emitAssign(AssignOpType op) {
  unsigned rank = emitNestedLoopHeader(op.getResult());
  indent();
  emitValue(op.getResult(), rank);
  os << " = ";
  emitValue(op.getOperand(), rank);
  os << ";";
  emitInfoAndNewLine(op);
  emitNestedLoopFooter(rank);
}

/// Control flow operation emitters.
void ModuleEmitter::emitCall(func::CallOp op) {
  // Handle returned value by the callee.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (mlir::isa<MemRefType>(result.getType()))
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  // Emit the function call.
  indent() << op.getCallee() << "(";

  // Handle input arguments.
  unsigned argIdx = 0;
  for (auto arg : op.getOperands()) {
    emitValue(arg);

    if (argIdx++ != op.getNumOperands() - 1)
      os << ", ";
  }

  // Handle output arguments.
  for (auto result : op.getResults()) {
    // The address should be passed in for scalar result arguments.
    if (mlir::isa<ShapedType>(result.getType()))
      os << ", ";
    else
      os << ", &";

    emitValue(result);
  }

  os << ");";
  emitInfoAndNewLine(op);
}

/// SCF statement emitters.
void ModuleEmitter::emitScfFor(scf::ForOp op) {
  for (auto [init, arg] : llvm::zip(op.getInitArgs(), op.getRegionIterArgs())) {
    indent();
    emitValue(arg);
    os << " = ";
    emitValue(init);
    os << ";";
    emitInfoAndNewLine(op);
  }
  indent() << "for (";
  auto iterVar = op.getInductionVar();

  // Emit lower bound.
  emitValue(iterVar);
  os << " = ";
  emitValue(op.getLowerBound());
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar);
  os << " < ";
  emitValue(op.getUpperBound());
  os << "; ";

  // Emit increase step.
  emitValue(iterVar);
  os << " += ";
  emitValue(op.getStep());
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();

  emitBlock(*op.getBody());
  reduceIndent();

  indent() << "}\n";

  for (auto [arg, result] :
       llvm::zip(op.getRegionIterArgs(), op.getResults())) {
    indent();
    emitValue(result);
    os << " = ";
    emitValue(arg);
    os << ";";
    emitInfoAndNewLine(op);
  }
}

void ModuleEmitter::emitScfIf(scf::IfOp op) {
  // Declare all values returned by scf::YieldOp. They will be further handled
  // by the scf::YieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (mlir::isa<MemRefType>(result.getType()))
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  indent() << "if (";
  emitValue(op.getCondition());
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();
  emitBlock(op.getThenRegion().front());
  reduceIndent();

  if (!op.getElseRegion().empty()) {
    indent() << "} else {\n";
    addIndent();
    emitBlock(op.getElseRegion().front());
    reduceIndent();
  }

  indent() << "}\n";
}

void ModuleEmitter::emitScfYield(scf::YieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  // For now, only and scf::If operations will use scf::Yield to return
  // generated values.
  if (auto parentOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHeader(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopFooter(rank);
    }
  } else if (auto forOp = dyn_cast<scf::ForOp>(op->getParentOp())) {
    for (auto [arg, yielded] :
         llvm::zip(forOp.getRegionIterArgs(), forOp.getYieldedValues())) {
      indent();
      emitValue(arg);
      os << " = ";
      emitValue(yielded);
      os << ";";
      emitInfoAndNewLine(op);
    }
  } else {
    llvm_unreachable("Unsupported this scf::YieldOp op");
  }
}

/// Affine statement emitters.
void ModuleEmitter::emitAffineFor(affine::AffineForOp op) {
  indent() << "for (";
  auto iterVar = op.getInductionVar();

  // Emit lower bound.
  emitValue(iterVar);
  os << " = ";
  auto lowerMap = op.getLowerBoundMap();
  AffineExprEmitter lowerEmitter(state, lowerMap.getNumDims(),
                                 op.getLowerBoundOperands());
  if (lowerMap.getNumResults() == 1)
    lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
  else {
    for (unsigned i = 0, e = lowerMap.getNumResults() - 1; i < e; ++i)
      os << "max(";
    lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
    for (auto &expr : llvm::drop_begin(lowerMap.getResults(), 1)) {
      os << ", ";
      lowerEmitter.emitAffineExpr(expr);
      os << ")";
    }
  }
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar);
  os << " < ";
  auto upperMap = op.getUpperBoundMap();
  AffineExprEmitter upperEmitter(state, upperMap.getNumDims(),
                                 op.getUpperBoundOperands());
  if (upperMap.getNumResults() == 1)
    upperEmitter.emitAffineExpr(upperMap.getResult(0));
  else {
    for (unsigned i = 0, e = upperMap.getNumResults() - 1; i < e; ++i)
      os << "min(";
    upperEmitter.emitAffineExpr(upperMap.getResult(0));
    for (auto &expr : llvm::drop_begin(upperMap.getResults(), 1)) {
      os << ", ";
      upperEmitter.emitAffineExpr(expr);
      os << ")";
    }
  }
  os << "; ";

  // Emit increase step.
  emitValue(iterVar);
  os << " += " << op.getStep() << ") {";
  emitInfoAndNewLine(op);

  addIndent();

  emitBlock(*op.getBody());
  reduceIndent();

  indent() << "}\n";
}

void ModuleEmitter::emitAffineIf(affine::AffineIfOp op) {
  // Declare all values returned by AffineYieldOp. They will be further
  // handled by the AffineYieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (mlir::isa<MemRefType>(result.getType()))
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  indent() << "if (";
  auto constrSet = op.getIntegerSet();
  AffineExprEmitter constrEmitter(state, constrSet.getNumDims(),
                                  op.getOperands());

  // Emit all constraints.
  unsigned constrIdx = 0;
  for (auto &expr : constrSet.getConstraints()) {
    constrEmitter.emitAffineExpr(expr);
    if (constrSet.isEq(constrIdx))
      os << " == 0";
    else
      os << " >= 0";

    if (constrIdx++ != constrSet.getNumConstraints() - 1)
      os << " && ";
  }
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();
  emitBlock(*op.getThenBlock());
  reduceIndent();

  if (op.hasElse()) {
    indent() << "} else {\n";
    addIndent();
    emitBlock(*op.getElseBlock());
    reduceIndent();
  }

  indent() << "}\n";
}

void ModuleEmitter::emitAffineParallel(affine::AffineParallelOp op) {
  // Declare all values returned by AffineParallelOp. They will be further
  // handled by the AffineYieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (mlir::isa<MemRefType>(result.getType()))
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  auto steps = getIntArrayAttrValue(op, op.getStepsAttrName());
  for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
    indent() << "for (";
    auto iterVar = op.getBody()->getArgument(i);

    // Emit lower bound.
    emitValue(iterVar);
    os << " = ";
    auto lowerMap = op.getLowerBoundsValueMap().getAffineMap();
    AffineExprEmitter lowerEmitter(state, lowerMap.getNumDims(),
                                   op.getLowerBoundsOperands());
    lowerEmitter.emitAffineExpr(lowerMap.getResult(i));
    os << "; ";

    // Emit upper bound.
    emitValue(iterVar);
    os << " < ";
    auto upperMap = op.getUpperBoundsValueMap().getAffineMap();
    AffineExprEmitter upperEmitter(state, upperMap.getNumDims(),
                                   op.getUpperBoundsOperands());
    upperEmitter.emitAffineExpr(upperMap.getResult(i));
    os << "; ";

    // Emit increase step.
    emitValue(iterVar);
    os << " += " << steps[i] << ") {";
    emitInfoAndNewLine(op);

    addIndent();
  }

  emitBlock(*op.getBody());

  for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
    reduceIndent();

    indent() << "}\n";
  }
}

void ModuleEmitter::emitAffineApply(affine::AffineApplyOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto affineMap = op.getAffineMap();
  AffineExprEmitter(state, affineMap.getNumDims(), op.getOperands())
      .emitAffineExpr(affineMap.getResult(0));
  os << ";";
  emitInfoAndNewLine(op);
}

template <typename OpType>
void ModuleEmitter::emitAffineMaxMin(OpType op, const char *syntax) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getOperands());
  for (unsigned i = 0, e = affineMap.getNumResults() - 1; i < e; ++i)
    os << syntax << "(";
  affineEmitter.emitAffineExpr(affineMap.getResult(0));
  for (auto &expr : llvm::drop_begin(affineMap.getResults(), 1)) {
    os << ", ";
    affineEmitter.emitAffineExpr(expr);
    os << ")";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitAffineLoad(affine::AffineLoadOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getMemRef());
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getMapOperands());
  for (auto index : affineMap.getResults()) {
    os << "[";
    affineEmitter.emitAffineExpr(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitAffineStore(affine::AffineStoreOp op) {
  indent();
  emitValue(op.getMemRef());
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getMapOperands());
  for (auto index : affineMap.getResults()) {
    os << "[";
    affineEmitter.emitAffineExpr(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  emitInfoAndNewLine(op);
}

// TODO: For now, all values created in the AffineIf region will be declared
// in the generated C++. However, values which will be returned by affine
// yield operation should not be declared again. How to "bind" the pair of
// values inside/outside of AffineIf region needs to be considered.
void ModuleEmitter::emitAffineYield(affine::AffineYieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  // For now, only AffineParallel and AffineIf operations will use
  // AffineYield to return generated values.
  if (auto parentOp = dyn_cast<affine::AffineIfOp>(op->getParentOp())) {
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHeader(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopFooter(rank);
    }
  } else if (auto parentOp =
                 dyn_cast<affine::AffineParallelOp>(op->getParentOp())) {
    indent() << "if (";
    unsigned ivIdx = 0;
    for (auto iv : parentOp.getBody()->getArguments()) {
      emitValue(iv);
      os << " == 0";
      if (ivIdx++ != parentOp.getBody()->getNumArguments() - 1)
        os << " && ";
    }
    os << ") {\n";

    // When all induction values are 0, generated values will be directly
    // assigned to the current results, correspondingly.
    addIndent();
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHeader(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopFooter(rank);
    }
    reduceIndent();

    indent() << "} else {\n";

    // Otherwise, generated values will be accumulated/reduced to the
    // current results with corresponding arith::AtomicRMWKind operations.

    indent() << "}\n";
  }
}

/// Helper to get the string indices of TransferRead/Write operations.
template <typename TransferOpType>
SmallVector<SmallString<8>, 4>
ModuleEmitter::getTransferIndices(TransferOpType op) {
  // Get the head indices of the transfer read/write.
  SmallVector<SmallString<8>, 4> indices;
  for (auto index : op.getIndices()) {
    assert(isDeclared(index) && "index has not been declared");
    indices.push_back(getName(index));
  }
  // Construct the physical indices.
  for (unsigned i = 0, e = op.getPermutationMap().getNumResults(); i < e; ++i) {
    auto expr = op.getPermutationMap().getResult(i);
    if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr))
      indices[dimExpr.getPosition()] += " + iv" + std::to_string(i);
  }
  return indices;
}

/// Helper to get the TransferRead/Write condition.
template <typename TransferOpType>
static SmallString<16>
getTransferCondition(TransferOpType op,
                     const SmallVector<SmallString<8>, 4> &indices) {
  // Figure out whether the transfer read/write could be out of bound.
  SmallVector<unsigned, 4> outOfBoundDims;
  for (unsigned i = 0, e = op.getVectorType().getRank(); i < e; ++i)
    if (!op.isDimInBounds(i))
      outOfBoundDims.push_back(i);

  // Construct the condition of transfer if required.
  SmallString<16> condition;
  for (auto i : outOfBoundDims) {
    auto expr = op.getPermutationMap().getResult(i);
    if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
      auto pos = dimExpr.getPosition();
      condition += indices[pos];
      condition += " < " + std::to_string(op.getShapedType().getDimSize(pos));
      if (i != outOfBoundDims.back())
        condition += " && ";
    }
  }
  return condition;
}

/// Memref-related statement emitters.
template <typename OpType> void ModuleEmitter::emitAlloc(OpType op) {
  // A declared result indicates that the memref is output of the function, and
  // has been declared in the function signature.
  if (isDeclared(op.getResult()))
    return;

  // Vivado HLS only supports static shape on-chip memory.
  if (!op.getType().hasStaticShape())
    emitError(op, "is unranked or has dynamic shape.");

  indent() << "__local ";
  emitArrayDecl(op.getResult());
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitLoad(memref::LoadOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getMemRef());
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitStore(memref::StoreOp op) {
  indent();
  emitValue(op.getMemRef());
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  emitInfoAndNewLine(op);
}

memref::SubViewOp getSubviewOp(Value val) {
  return val.getDefiningOp<memref::SubViewOp>();
}

bool checkOneDimMemref(Value val) {
  if (auto memrefTy = mlir::dyn_cast<MemRefType>(val.getType())) {
    return memrefTy.getRank() == 1;
  }
  return false;
}

bool checkSubViewOffsetAndStride(memref::SubViewOp op) {
  for (auto off : op.getMixedOffsets()) {
    if (!isConstantIntValue(off, 0)) {
      return false;
    }
  }
  for (auto stride : op.getMixedStrides()) {
    if (!isConstantIntValue(stride, 1)) {
      return false;
    }
  }
  return true;
}

bool is1D(MemRefType memref) { return memref.getRank() == 1; }

bool is2D(MemRefType memref) { return memref.getRank() == 2; }

bool lessOrEqual2D(MemRefType memref) { return memref.getRank() <= 2; }

bool isSimilar1D(MemRefType memref) {
  for (int i = 0; i < memref.getRank() - 1; i++)
    if (memref.getShape()[i] != 1)
      return false;
  return true;
}

void ModuleEmitter::emitMemCpyValue(Value val) {
  auto memrefType = mlir::cast<MemRefType>(val.getType());
  if (auto castOp = val.getDefiningOp<memref::ReinterpretCastOp>()) {
    emitValue(castOp.getSource());
    os << " + ";
    emitOpFoldResult(castOp.getMixedOffsets()[0]);
    if (!isSimilar1D(memrefType)) {
      os << " + i * ";
      emitOpFoldResult(castOp.getMixedStrides()[0]);
    }
  } else if (auto allocOp = val.getDefiningOp<memref::AllocOp>()) {
    emitValue(val);
    if (!isSimilar1D(memrefType)) {
      os << " + i";
    }
  } else if (auto blockArg = mlir::dyn_cast<BlockArgument>(val)) {
    emitValue(val);
  } else {
    val.dump();
    llvm_unreachable("mecpy unsupported subview targetOp");
  }
}

void ModuleEmitter::emitOpFoldResult(OpFoldResult opFoldResult) {
  if (auto intAttr = getConstantIntValue(opFoldResult)) {
    os << intAttr;
  } else {
    emitValue(mlir::dyn_cast<Value>(opFoldResult));
  }
}

void ModuleEmitter::emitAsyncCopy(Value target, Value source) {
  indent() << "ev = async_work_group_copy(";
  emitMemCpyValue(target);
  os << ", ";
  emitMemCpyValue(source);
  os << ", ";
}

void ModuleEmitter::emitAsyncCopyWithOpFoldResult(Value target, Value source,
                                                  OpFoldResult num) {
  emitAsyncCopy(target, source);
  emitOpFoldResult(num);
  os << ", 0);";
}

void ModuleEmitter::emitAsyncCopyWithConstant(Value target, Value source,
                                              int num) {
  emitAsyncCopy(target, source);
  os << num << ", 0);";
}

void ModuleEmitter::emitMemCpy(memref::CopyOp op) {
  auto sourceSubView = getSubviewOp(op.getSource());
  auto targetSubView = getSubviewOp(op.getTarget());
  bool isCopySubView = false;
  if (sourceSubView || targetSubView) {
    assert(checkSubViewOffsetAndStride(sourceSubView) &&
           "source subview not support");
    assert(checkSubViewOffsetAndStride(targetSubView) &&
           "target subview not support");
    isCopySubView = true;
  }

  auto targetMemref =
      mlir::dyn_cast<mlir::MemRefType>(op.getTarget().getType());
  assert(lessOrEqual2D(targetMemref) && "mecpy unsupported not support > 2D");
  if (isCopySubView) {
    if (is1D(targetMemref)) {
      emitAsyncCopyWithOpFoldResult(targetSubView.getSource(),
                                    sourceSubView.getSource(),
                                    targetSubView.getMixedSizes()[0]);
    } else if (is2D(targetMemref)) {
      indent() << "for (int i = 0; i < ";
      emitOpFoldResult(targetSubView.getMixedSizes()[0]);
      os << "; i += 1) {\n";
      addIndent();
      emitAsyncCopyWithOpFoldResult(targetSubView.getSource(),
                                    sourceSubView.getSource(),
                                    targetSubView.getMixedSizes()[1]);
      os << "\n";
      reduceIndent();
      indent() << "}";
    }
  } else {
    if (isSimilar1D(targetMemref)) {
      emitAsyncCopyWithConstant(op.getTarget(), op.getSource(),
                                targetMemref.getNumElements());
    } else if (is2D(targetMemref)) {
      if (auto castOp =
              op.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
        indent() << "for (int i = 0; i < ";
        emitOpFoldResult(castOp.getMixedSizes()[0]);
        os << "; i += 1) {\n";
        addIndent();
        emitAsyncCopyWithOpFoldResult(op.getTarget(), op.getSource(),
                                      castOp.getMixedSizes()[1]);
        os << "\n";
        reduceIndent();
        indent() << "}";
      }
    }
  }
  emitInfoAndNewLine(op);
  indent() << "wait_group_events(1, &ev);\n";
}

template <typename OpType> void ModuleEmitter::emitReshape(OpType op) {
  auto array = op->getResult(0);
  assert(!isDeclared(array) && "has been declared before.");

  auto arrayType = mlir::cast<ShapedType>(array.getType());
  indent() << arrayType << " (*";

  // Add the new value to nameTable and emit its name.
  os << addName(array, false);
  os << ")";

  for (auto &shape : llvm::drop_begin(arrayType.getShape(), 1))
    os << "[" << shape << "]";

  os << " = (" << arrayType << "(*)";
  for (auto &shape : llvm::drop_begin(arrayType.getShape(), 1))
    os << "[" << shape << "]";
  os << ") ";

  emitValue(op->getOperand(0));
  os << ";";
  emitInfoAndNewLine(op);
}

/// Standard expression emitters.
void ModuleEmitter::emitUnary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHeader(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  os << " = " << syntax << "(";
  emitValue(op->getOperand(0), rank);
  os << ");";
  emitInfoAndNewLine(op);
  emitNestedLoopFooter(rank);
}

void ModuleEmitter::emitBinary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHeader(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  os << " = ";
  emitValue(op->getOperand(0), rank);
  os << " " << syntax << " ";
  emitValue(op->getOperand(1), rank);
  os << ";";
  emitInfoAndNewLine(op);
  emitNestedLoopFooter(rank);
}

template <typename OpType>
void ModuleEmitter::emitMaxMin(OpType op, const char *syntax) {
  auto rank = emitNestedLoopHeader(op.getResult());
  indent();
  emitValue(op.getResult());
  os << " = " << syntax << "(";
  emitValue(op.getLhs(), rank);
  os << ", ";
  emitValue(op.getRhs(), rank);
  os << ");";
  emitInfoAndNewLine(op);
  emitNestedLoopFooter(rank);
}

/// Special expression emitters.
void ModuleEmitter::emitSelect(arith::SelectOp op) {
  unsigned rank = emitNestedLoopHeader(op.getResult());
  unsigned conditionRank = rank;
  if (!mlir::isa<ShapedType>(op.getCondition().getType()))
    conditionRank = 0;

  indent();
  emitValue(op.getResult(), rank);
  os << " = ";
  emitValue(op.getCondition(), conditionRank);
  os << " ? ";
  // os << "(" << getTypeName(op.getTrueValue()) << ")";
  emitValue(op.getTrueValue(), rank);
  os << " : ";
  // os << "(" << getTypeName(op.getFalseValue()) << ")";
  emitValue(op.getFalseValue(), rank);
  os << ";";
  emitInfoAndNewLine(op);
  emitNestedLoopFooter(rank);
}

template <typename OpType> void ModuleEmitter::emitConstant(OpType op) {
  // This indicates the constant type is scalar (float, integer, or bool).
  if (isDeclared(op.getResult()))
    return;

  if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(op.getValue())) {
    indent();
    emitArrayDecl(op.getResult());
    os << " = {";
    auto type =
        mlir::cast<MemRefType>(op.getResult().getType()).getElementType();

    unsigned elementIdx = 0;
    for (auto element : denseAttr.template getValues<Attribute>()) {
      auto string = getConstantString(type, element);
      if (string.empty())
        op.emitOpError("constant has invalid value");
      os << string;
      if (elementIdx++ != denseAttr.getNumElements() - 1)
        os << ", ";
    }
    os << "};";
    emitInfoAndNewLine(op);
  } else
    emitError(op, "has unsupported constant type.");
}

void ModuleEmitter::emitFunction(func::FuncOp func) {
  if (func.getBlocks().size() != 1)
    emitError(func, "has zero or more than one basic blocks.");

  // Emit function signature.
  os << "__kernel void " << func.getName() << "(\n";
  addIndent();

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit input arguments.
  unsigned argIdx = 0;
  for (auto &arg : func.getArguments()) {
    indent();
    auto type = arg.getType();

    if (mlir::isa<MemRefType>(type))
      emitArrayDecl(arg);
    else
      emitValue(arg);

    portList.push_back(arg);
    if (argIdx++ != func.getNumArguments() - 1)
      os << ",\n";
  }

  reduceIndent();
  os << "\n) {";
  emitInfoAndNewLine(func);

  // Emit function body.
  addIndent();
  // Emit event
  indent() << "event_t ev = 0;\n";

  emitBlock(func.front());
  reduceIndent();
  os << "}\n";
  // An empty line.
  os << "\n";
}

void ModuleEmitter::emitGlobalId(gpu::GlobalIdOp op) {
  indent();
  emitValue(op.getResult());
  os << " = get_global_id(";
  gpu::Dimension dim = op.getDimension();
  os << static_cast<int64_t>(dim);
  os << ");\n";
}

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  os << R"XXX(//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for OpenCL
//
//===----------------------------------------------------------------------===//

)XXX";

  // Emit all functions in the call graph in a post order.
  CallGraph graph(module);
  llvm::SmallDenseSet<func::FuncOp> emittedFuncs;
  for (auto node : llvm::post_order<const CallGraph *>(&graph)) {
    if (node->isExternal())
      continue;
    if (auto func =
            node->getCallableRegion()->getParentOfType<func::FuncOp>()) {
      emitFunction(func);
      emittedFuncs.insert(func);
    }
  }

  // Emit remained functions accordingly.
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      if (!emittedFuncs.count(func))
        emitFunction(func);
    } else if (!isa<ml_program::GlobalOp>(op))
      emitError(&op, "is unsupported operation");
  }
}

//===----------------------------------------------------------------------===//
// Entry of triton-spirv-translate
//===----------------------------------------------------------------------===//

LogicalResult emitOpenCL(ModuleOp module, llvm::raw_ostream &os) {
  ScaleHLSEmitterState state(os);
  ModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void registerEmitOpenCLTranslation() {
  static TranslateFromMLIRRegistration toOpenCL(
      "triton-spirv-emit-opencl", "Translate MLIR into OpenCL", emitOpenCL,
      [&](DialectRegistry &registry) {
        registry.insert<mlir::math::MathDialect, mlir::arith::ArithDialect,
                        mlir::scf::SCFDialect, mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect, ::mlir::gpu::GPUDialect,
                        mlir::affine::AffineDialect>();
      });
}

int main(int argc, char **argv) {
  registerEmitOpenCLTranslation();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "SPIRV Translation Tool"));
}