/// Implements the Cinm dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Conversion/CommonPatterns.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"

#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "cinm-ops"

using namespace mlir;
using namespace mlir::cinm;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Cinm/IR/CinmEnums.cpp.inc"

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CinmDialect
//===----------------------------------------------------------------------===//

void CinmDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.cpp.inc"
      >();
}

namespace mlir {
namespace cinm {

cinm::ComputeOp getEnclosingComputeBlock(Operation *op) {
  Operation *parent = op;
  while ((parent = parent->getParentOp())) {
    if (auto parentCompute = dyn_cast<cinm::ComputeOp>(parent))
      return parentCompute;
  }

  assert(false && "CINM operator is not inside a cinm.compute block");
}

SmallVector<Value> MinOp::convertToTiledOps(OpBuilder builder,
                                            TilingParameters params) {
  auto ty = getInput().getType();
  return {createVectorReduceMin(
      builder, getLoc(), getOperand(),
      params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType()))};
}

SmallVector<Value> MaxOp::convertToTiledOps(OpBuilder builder,
                                            TilingParameters params) {
  auto ty = getInput().getType();
  return {createVectorReduceMax(
      builder, getLoc(), getOperand(),
      params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType()))};
}

/** This tiling works this way:
    - Given a Gemm of dimensions (%A: <MxR>), (%B: <RxN>) -> <MxN>
    - If M == 1 then
      - determine optimal tile sizes RT and NT for the R and N dimensions
      - then emit the following program:
        affine.loop %j = 0 to N step NT iter_args(%acc0 = empty: <1xN>){
          %tile: <1xNT> = affine.loop %k = 0 to R step RT iter_args(%acc =
   zeros: <1xNT>) { %rowTile = slice %A[0, %k] [1, RT] [1, 1] : <1xRT> %colTile
   = slice %B[%k, %j] [RT, NT] [1, 1] : <RTxNT> %tmp = cinm.gemm %rowTile,
   %colTile : -> <1xNT> %add = cinm.add %acc, tmp affine.yield %add : <1xNT>
          }
          %tmp = insert_slice %tile, %acc0[0, %j] [1, NT] [1, 1]
          affine.yield %tmp
        }
    - if M > 1 then the gemm is first reduced into a loop over M, and gemms of
   size <1xR> <RxN>

 */
SmallVector<Value> GemmOp::convertToTiledOps(OpBuilder builder,
                                             TilingParameters params) {
  Value lhs = getLeft();
  Value rhs = getRight();

  const RankedTensorType lhsType = getLeft().getType();
  const RankedTensorType rhsType = getRight().getType();

  const RankedTensorType resultType =
      getResult().getType().cast<RankedTensorType>();
  const ArrayRef<int64_t> resultShape = resultType.getShape();

  const Value resultInit = builder.create<tensor::EmptyOp>(
      getLoc(), resultShape, resultType.getElementType());

  // Size of the tile on the reduction dimension.
  auto reduceClusterSize = params.reduceClusterSize(
      2, resultType.getNumElements(), lhsType.getElementType());
  auto parallelTileSize =
      params.parallelClusterSize(rhsType.getDimSize(1), reduceClusterSize);

  if (lhsType.getDimSize(0) == 1) {
    // then it's basically vector-mat multiplication
    // arguments have shape <1xR>, <RxB>
    auto r = lhsType.getDimSize(1);
    auto b = rhsType.getDimSize(1);
    auto eltTy = rhsType.getElementType();

    // iterate over b (parallel loop)
    return createNestedAffineForLoops(
        builder, getLoc(), {b}, {parallelTileSize}, {resultInit},
        [&](OpBuilder &builder, Location loc, ValueRange indices,
            ValueRange iterArgs) -> SmallVector<Value> {
          auto reductionAccTy =
              RankedTensorType::get({1, parallelTileSize}, eltTy);
          auto cst0 = builder.create<arith::ConstantOp>(
              loc, builder.getZeroAttr(eltTy));
          auto reductionAccInit =
              builder.create<tensor::SplatOp>(loc, cst0, reductionAccTy);

          const auto indexInParDim = indices[0];

          // this is the reduction loop
          SmallVector<Value, 1> reductionResult = createNestedAffineForLoops(
              builder, loc, {r}, {reduceClusterSize},
              reductionAccInit->getResults(),
              [&](OpBuilder &builder, Location loc, ValueRange indices,
                  ValueRange iterArgs) -> SmallVector<Value> {
                const auto indexInRedDim = indices[0];

                const ArrayRef<int64_t> lhsOffsets{0, ShapedType::kDynamic};
                const ArrayRef<int64_t> lhsSizes{1, reduceClusterSize};
                const ArrayRef<int64_t> lhsStrides{1, 1};

                const Type lhsSliceType =
                    RankedTensorType::get({1, reduceClusterSize}, eltTy);

                const ArrayRef<Value> lhsDynamicOffsets{indexInRedDim};
                const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
                    loc, lhsSliceType, lhs, lhsDynamicOffsets, ValueRange{},
                    ValueRange{}, lhsOffsets, lhsSizes, lhsStrides);

                // todo this is still a square tile but the left tile is flat
                const Type rhsSliceType = RankedTensorType::get(
                    {reduceClusterSize, parallelTileSize}, eltTy);

                const ArrayRef<int64_t> rhsOffsets{ShapedType::kDynamic,
                                                   ShapedType::kDynamic};
                const ArrayRef<int64_t> rhsSizes{reduceClusterSize,
                                                 parallelTileSize};
                const ArrayRef<int64_t> rhsStrides{1, 1};

                const Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
                    loc, rhsSliceType, rhs,
                    ValueRange{indexInParDim, indexInRedDim}, ValueRange{},
                    ValueRange{}, rhsOffsets, rhsSizes, rhsStrides);

                // now we have a LHS tile <reduceClusterSize>
                // and RHS tile <reduceClusterSize x parallelTileSize

                // Here we're back to doing
                // GEMM(ltile: <1 x rcs>, rtile: <rcs x pts>) + iterArgs[0]
                auto tmpReduce = builder.create<cinm::GemmOp>(
                    loc, lhsSlice, rhsSlice, iterArgs[0]);
                cinm::markOpAsNoTile(tmpReduce);
                return {tmpReduce};
              });

          const ArrayRef<int64_t> resultOffsets{0, ShapedType::kDynamic};
          const ArrayRef<int64_t> &resultSizes{1, parallelTileSize};
          const ArrayRef<int64_t> resultStrides{1, 1};
          const ValueRange resultDynamicOffsets{indexInParDim};

          const Value result = builder.create<tensor::InsertSliceOp>(
              loc, reductionResult[0], iterArgs[0], resultDynamicOffsets,
              ValueRange{}, ValueRange{}, resultOffsets, resultSizes,
              resultStrides);

          return {result};
        });
  }

  const ArrayRef<int64_t> tileSizes = {1, parallelTileSize};

  return createNestedAffineForLoops(
      builder, getLoc(), resultShape, tileSizes, ValueRange{resultInit},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        const ArrayRef<int64_t> lhsOffsets{ShapedType::kDynamic, 0};
        const ArrayRef<int64_t> lhsSizes{tileSizes[0], lhsType.getDimSize(1)};
        const ArrayRef<int64_t> unitStrides{1, 1};
        const ArrayRef<int64_t> &lhsStrides = unitStrides;
        const ArrayRef<int64_t> rhsOffsets{0, ShapedType::kDynamic};
        const ArrayRef<int64_t> rhsSizes{rhsType.getDimSize(0), tileSizes[1]};
        const ArrayRef<int64_t> &rhsStrides = unitStrides;
        const ArrayRef<int64_t> resultOffsets{ShapedType::kDynamic,
                                              ShapedType::kDynamic};
        const ArrayRef<int64_t> &resultSizes = tileSizes;
        const ArrayRef<int64_t> resultStrides = unitStrides;

        const SmallVector<Value> lhsDynamicOffsets{indices[0]};
        const RankedTensorType lhsSliceType =
            tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                2, lhsType, lhsOffsets, lhsSizes, lhsStrides);
        const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, lhsSliceType, lhs, lhsDynamicOffsets, ValueRange{},
            ValueRange{}, lhsOffsets, lhsSizes, lhsStrides);

        const SmallVector<Value> rhsDynamicOffsets{indices[1]};
        const Type rhsSliceType =
            tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                2, rhsType, rhsOffsets, rhsSizes, rhsStrides);
        const Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, rhsSliceType, rhs, rhsDynamicOffsets, ValueRange{},
            ValueRange{}, rhsOffsets, rhsSizes, rhsStrides);

        GemmOp smallerGemm =
            builder.create<cinm::GemmOp>(loc, lhsSlice, rhsSlice);
        // may be tiled further
        // cinm::markOpAsNoTile(smallerGemm);

        const Value result = builder.create<tensor::InsertSliceOp>(
            loc, smallerGemm.getResult(), iterArgs[0], indices, ValueRange{},
            ValueRange{}, resultOffsets, resultSizes, resultStrides);

        return {result};
      });
}

::mlir::LogicalResult GemmOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,
    GemmOp::Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getLeft().getType());
  ShapeAdaptor rhsShape(adaptor.getRight().getType());

  if (lhsShape.getRank() == 2 && rhsShape.getRank() == 2 &&
      lhsShape.getDimSize(1) == rhsShape.getDimSize(0) &&
      lhsShape.getElementType() == rhsShape.getElementType()) {

    SmallVector<int64_t, 2> outShape;
    outShape.push_back(lhsShape.getDimSize(0));
    outShape.push_back(rhsShape.getDimSize(1));

    inferredReturnShapes.push_back(
        ShapedTypeComponents(outShape, lhsShape.getElementType()));
    return success();
  }
  return failure();
  //  return context->emitError("operand types are not compatible");
}

SmallVector<Value> GemvOp::convertToTiledOps(OpBuilder builder,
                                             TilingParameters params) {
  const Value lhs = getOperand(0);
  const Value rhs = getOperand(1);

  const RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();
  const auto lhsShape = lhsType.getShape();

  const RankedTensorType resultType =
      getResult().getType().cast<RankedTensorType>();
  const ArrayRef<int64_t> resultShape = resultType.getShape();

  const Value resultInit = builder.create<tensor::EmptyOp>(
      getLoc(), resultShape, resultType.getElementType());
  // Size of the tile on the reduction dimension.
  auto reduceClusterSize = params.reduceClusterSize(
      2, resultType.getNumElements(), lhsType.getElementType());
  auto parallelTileSize =
      params.parallelClusterSize(lhsShape[0], reduceClusterSize);

  return createNestedAffineForLoops(
      builder, getLoc(), {lhsShape[0]}, {parallelTileSize},
      ValueRange{resultInit},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        const SmallVector<int64_t, 2> lhsOffsets{ShapedType::kDynamic, 0};
        const SmallVector<int64_t, 2> lhsSizes{parallelTileSize, lhsShape[1]};
        const SmallVector<int64_t, 2> lhsStrides{1, 1};

        const SmallVector<int64_t, 1> resultOffsets{ShapedType::kDynamic};
        const SmallVector<int64_t, 1> resultSizes{parallelTileSize};
        const SmallVector<int64_t, 1> resultStrides{1};

        const RankedTensorType lhsSliceType =
            tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                1, lhsType, lhsOffsets, lhsSizes, lhsStrides);
        const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, lhsSliceType, lhs, indices, ValueRange{}, ValueRange{},
            lhsOffsets, lhsSizes, lhsStrides);

        // todo should be cinm.reduce (or linalg.reduce)
        auto mult = builder.create<cinm::MulOp>(loc, lhsSlice, rhs);
        auto add = builder.create<cinm::AddOp>(loc, mult, iterArgs[0]);
        markOpAsNoTile(mult);
        markOpAsNoTile(add);

        const Value result = builder.create<tensor::InsertSliceOp>(
            loc, add.getResult(), iterArgs[0], indices, ValueRange{},
            ValueRange{}, resultOffsets, resultSizes, resultStrides);

        return {result};
      });
}

template <typename OP>
SmallVector<Value> tileElementWiseBinaryOp(ImplicitLocOpBuilder builder, OP op,
                                           TilingParameters params) {
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);

  RankedTensorType tensorTy = lhs.getType().cast<RankedTensorType>();
  if (params.workingGroupSize() > tensorTy.getNumElements()) {
    // this is dead code as the op is marked dynamically legal in this case
    assert(false && "Working group is too large");
  }

  auto reduceClusterSize = params.reduceClusterSize(
      3, tensorTy.getNumElements(), tensorTy.getElementType());
  reduceClusterSize *= params.workingGroupSize();
  assert(reduceClusterSize <= tensorTy.getNumElements());

  auto shape = tensorTy.getShape();
  const RankedTensorType originalType = tensorTy;
  Value originalShapeValue;
  if (shape.size() > 1) {
    // then flatten the tensors first
    originalShapeValue = builder.create<arith::ConstantOp>(
        RankedTensorType::get({static_cast<int64_t>(shape.size())},
                              builder.getI64Type()),
        builder.getI64TensorAttr(shape));
    lhs = cinm::reshapeStatic(builder, builder.getLoc(), op.getLhs(),
                              {tensorTy.getNumElements()});
    rhs = cinm::reshapeStatic(builder, builder.getLoc(), op.getRhs(),
                              {tensorTy.getNumElements()});
    tensorTy = lhs.getType().cast<RankedTensorType>();
    shape = tensorTy.getShape();
  }

  const Value resultInit =
      builder.create<tensor::EmptyOp>(tensorTy, ValueRange{});

  const SmallVector<int64_t, 2> tileSizes = {
      tensorTy.getNumElements() / reduceClusterSize, reduceClusterSize};

  SmallVector<Value, 1> result = createNestedAffineForLoops(
      builder, op.getLoc(), {tensorTy.getNumElements()}, {reduceClusterSize},
      ValueRange{resultInit},
      [&](OpBuilder &builder, Location loc, ValueRange indices,
          ValueRange iterArgs) -> SmallVector<Value> {
        const SmallVector<int64_t, 2> offsets{ShapedType::kDynamic};
        const SmallVector<int64_t, 2> sizes{reduceClusterSize};
        const SmallVector<int64_t, 2> strides{1};

        const RankedTensorType sliceTy = RankedTensorType::get(
            {reduceClusterSize}, tensorTy.getElementType());

        const Value lhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, sliceTy, lhs, indices, ValueRange{}, ValueRange{}, offsets,
            sizes, strides);
        const Value rhsSlice = builder.create<tensor::ExtractSliceOp>(
            loc, sliceTy, rhs, indices, ValueRange{}, ValueRange{}, offsets,
            sizes, strides);

        // here we recreate the same op with smaller dimensions
        OP smaller = builder.create<OP>(loc, lhsSlice, rhsSlice);
        markOpAsNoTile(smaller);

        const Value result = builder.create<tensor::InsertSliceOp>(
            loc, smaller.getResult(), iterArgs[0], indices, ValueRange{},
            ValueRange{}, offsets, sizes, strides);

        return {result};
      });
  if (originalType.getShape().size() > 1) {
    result[0] = builder.create<tensor::ReshapeOp>(originalType, result[0],
                                                  originalShapeValue);
  }
  return result;
}

SmallVector<Value> AddOp::convertToTiledOps(OpBuilder builder,
                                            TilingParameters params) {
  ImplicitLocOpBuilder ibuilder(getLoc(), builder);
  return tileElementWiseBinaryOp<AddOp>(ibuilder, *this, params);
}

::mlir::LogicalResult GemvOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, std::optional<::mlir::Location>, Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  // todo verify sizes

  auto result = ShapedTypeComponents(adaptor.getRight().getType());
  inferredReturnShapes.emplace_back(std::move(result));
  return success();
}

::mlir::LogicalResult SimSearchOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, std::optional<::mlir::Location>, Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {

  ShapeAdaptor inputShape(adaptor.getLeft().getType());
  auto elt = inputShape.getElementType();

  SmallVector<int64_t> outputShape;
  outputShape.resize(1, ShapedType::kDynamic);
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  return success();
}

::mlir::LogicalResult TopKOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, std::optional<::mlir::Location>, Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {

  ShapeAdaptor inputShape(adaptor.getInput().getType());
  auto elt = inputShape.getElementType();

  SmallVector<int64_t> outputShape;
  outputShape.resize(1, ShapedType::kDynamic);
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, elt));
  return success();
}

// Copied from the TOSA codebase.
::mlir::LogicalResult TransposeOp::inferReturnTypeComponents(
    ::mlir::MLIRContext *, std::optional<::mlir::Location>, Adaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>
        &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  ShapeAdaptor permsShape(adaptor.getPerms().getType());

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputShape.hasRank() || !permsShape.hasRank() ||
      permsShape.isDynamicDim(0)) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // This would imply the number of permutations does not match the rank of the
  // input which is illegal.
  if (permsShape.getDimSize(0) != inputShape.getRank()) {
    return failure();
  }

  // Without the input dims we cannot determine the output dim sizes but we
  // can determine the output rank.
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(permsShape.getDimSize(0), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Rank-0 means no permutations matter.
  if (inputShape.getRank() == 0) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Check whether the input dimensions are all the same.
  bool allTheSame = true;
  for (int i = 1, s = inputShape.getRank(); i < s; i++) {
    if (inputShape.getDimSize(0) != inputShape.getDimSize(i)) {
      allTheSame = false;
      break;
    }
  }

  // If all of the input dimensions are the same we don't care about the
  // permutation.
  if (allTheSame) {
    outputShape.resize(inputShape.getRank(), inputShape.getDimSize(0));
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  outputShape.resize(inputShape.getRank(), ShapedType::kDynamic);
  // If the permuations are a constant we can directly determine the output
  // shape.
  DenseIntElementsAttr attr;
  if (matchPattern(adaptor.getPerms(), m_Constant(&attr)) &&
      attr.getType().getRank() == 1) {
    ShapeAdaptor permShape = attr;
    outputShape.reserve(inputShape.getRank());
    for (int i = 0, s = inputShape.getRank(); i < s; i++) {
      outputShape[i] = inputShape.getDimSize(permShape.getDimSize(i));
    }
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

SmallVector<Value> ReduceOp::convertToTiledOps(OpBuilder builder,
                                               TilingParameters params) {
  auto ty = getInput().getType();
  auto reduceClusterSize =
      params.reduceClusterSize(1, ty.getNumElements(), ty.getElementType());
  if (getMethod() == ReduceMethod::ADD) {
    return {createVectorReduceAdd(builder, getLoc(), getOperand(),
                                  reduceClusterSize)};
  } else if (getMethod() == ReduceMethod::MUL) {
    return {createVectorReduceMul(builder, getLoc(), getOperand(),
                                  reduceClusterSize)};
  } else {
    abort();
  }
}

ParseResult SimSearchOp::parse(::mlir::OpAsmParser &parser,
                               ::mlir::OperationState &result) {

  //  let assemblyFormat= "$metric `,` $k `(` $left `,` $right `)` attr-dict `:`
  //  type($left)";
  std::string opname;
  if (parser.parseKeywordOrString(&opname))
    return failure();

  SimilarityMetric metric;
  if (opname == "cos") {
    metric = SimilarityMetric::COS;
  } else if (opname == "dot") {
    metric = SimilarityMetric::DOT;
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected a string \"cos\" or \"dot\"");
  }

  result.addAttribute(getMetricAttrName(result.name),
                      SimilarityMetricAttr::get(result.getContext(), metric));

  int64_t numK;
  if (parser.parseInteger(numK))
    return failure();

  auto i64Ty = IntegerType::get(result.getContext(), 64);
  result.addAttribute(getKAttrName(result.name), IntegerAttr::get(i64Ty, numK));

  OpAsmParser::UnresolvedOperand left, right;
  NamedAttrList attrDict;
  Type opTy;

  if (parser.parseLParen() || parser.parseOperand(left, false) ||
      parser.parseComma() || parser.parseOperand(right, false) ||
      parser.parseRParen() || parser.parseColonType(opTy))
    return failure();

  result.addAttributes(attrDict);
  if (parser.resolveOperand(left, opTy, result.operands) ||
      parser.resolveOperand(right, opTy, result.operands))
    return failure();

  // finally add result types

  auto eltTy = opTy.cast<RankedTensorType>().getElementType();
  auto resultValuesTy = RankedTensorType::get({ShapedType::kDynamic}, eltTy);
  auto resultIndicesTy = RankedTensorType::get(
      {ShapedType::kDynamic}, IndexType::get(result.getContext()));
  result.addTypes({resultValuesTy, resultIndicesTy});

  return success();
}

void SimSearchOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printKeywordOrString(stringifySimilarityMetric(getMetric()));
  p << ' ' << getK().getInt();
  p << " (";
  p.printOperand(getLeft());
  p << ", ";
  p.printOperand(getRight());
  p << ") : ";
  p.printType(getLeft().getType());
}

} // namespace cinm
} // namespace mlir

// parsers/printers
