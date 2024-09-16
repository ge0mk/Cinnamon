#include "cinm-mlir/Dialect/Cinm/IR/CinmBase.h"
#include "cinm-mlir/Dialect/Cinm/IR/CinmOps.h"
#include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.h"
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h"

#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Affine/Analysis/Utils.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopFusionUtils.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Transform/IR/TransformInterfaces.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::cinm {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_CINMLOOPPERMUTATIONPASS
#define GEN_PASS_DEF_CINMLOOPFUSIONPASS
#include "cinm-mlir/Dialect/Cinm/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

void permuteLoops(MutableArrayRef<affine::AffineForOp> inputNest,
                  ArrayRef<unsigned int> permMap) {
  affine::permuteLoops(inputNest, permMap);
  // affine::permuteLoops doesn't update iter inits / yielded values so we do

  // replace old inner body parameters with new ones
  affine::AffineForOp oldInnerLoop = inputNest.back();
  affine::AffineForOp newInnerLoop = inputNest[permMap.back()];
  for (size_t i = 0; i < oldInnerLoop.getNumResults(); i++) {
    oldInnerLoop.getBody()->getArgument(i + 1).replaceAllUsesWith(
        newInnerLoop.getBody()->getArgument(i + 1));
  }

  // replace yield values of new inner loop with old inner loop
  affine::AffineYieldOp oldInnerYield =
      cast<affine::AffineYieldOp>(oldInnerLoop.getBody()->back());
  affine::AffineYieldOp newInnerYield =
      cast<affine::AffineYieldOp>(newInnerLoop.getBody()->back());
  for (size_t i = 0; i < oldInnerYield.getNumOperands(); i++) {
    newInnerYield.setOperand(i, oldInnerYield.getOperand(i));
  }

  // replace inits of new outer loop with old outer loop
  affine::AffineForOp oldOuterLoop = inputNest.front();
  affine::AffineForOp newOuterLoop = inputNest[permMap.front()];
  for (size_t i = 0; i < oldOuterLoop.getInits().size(); i++) {
    newOuterLoop.getInitsMutable()[i].assign(oldOuterLoop.getInits()[i]);
  }

  // replace results of old outer loop with new outer loop
  for (size_t i = 0; i < inputNest.front().getNumResults(); i++) {
    inputNest.front().getResult(i).replaceAllUsesWith(
        inputNest[permMap.front()].getResult(i));
  }

  // replace yield values of outer loops with result values of the loop before
  // the yield op & replace nested loop inits with parent loop inits
  for (size_t i = 0; i < permMap.size() - 1; i++) {
    affine::AffineForOp outer = inputNest[permMap[i]];
    affine::AffineForOp inner =
        dyn_cast<affine::AffineForOp>(outer.getBody()->front());
    affine::AffineYieldOp yield =
        dyn_cast<affine::AffineYieldOp>(outer.getBody()->back());

    for (size_t k = 0; k < inner.getNumResults(); k++) {
      yield.setOperand(k, inner.getResult(k));
      inner.getInitsMutable()[k].assign(outer.getBody()->getArgument(k + 1));
    }
  }
}

void fuseLoops(MutableArrayRef<affine::AffineForOp> loops) {
  if (loops.size() <= 1) {
    return;
  }

  SmallVector<Value> inits;
  for (auto &loop : loops) {
    inits.append(loop.getInits().begin(), loop.getInits().end());
  }

  OpBuilder builder{loops.front().getContext()};
  builder.setInsertionPointAfter(loops.back());
  affine::AffineForOp result = builder.create<affine::AffineForOp>(
      loops.front().getLoc(), loops.front().getConstantLowerBound(),
      loops.front().getConstantUpperBound(), loops.front().getStepAsInt(),
      inits);

  auto &resultBody = result.getBody()->getOperations();
  SmallVector<Value> yieldValues;
  size_t k = 0;
  for (auto &loop : loops) {
    auto &body = loop.getBody()->getOperations();
    resultBody.splice(resultBody.end(), body, body.begin(),
                      std::prev(body.end()));

    if (auto yieldOp = dyn_cast<affine::AffineYieldOp>(body.back())) {
      for (size_t i = 0; i < yieldOp.getNumOperands(); i++) {
        yieldValues.push_back(yieldOp.getOperand(i));
      }
    }

    loop.getBody()->getArgument(0).replaceAllUsesWith(
        result.getBody()->getArgument(0));

    for (size_t i = 0; i < loop.getNumIterOperands(); i++, k++) {
      loop.getResult(i).replaceAllUsesWith(result.getResult(k));
      loop.getBody()->getArgument(i + 1).replaceAllUsesWith(
          result.getBody()->getArgument(k + 1));
    }
  }

  builder.setInsertionPointToEnd(result.getBody());
  builder.create<affine::AffineYieldOp>(
      loops.front().getBody()->back().getLoc(), yieldValues);

  for (auto &loop : loops) {
    loop.erase();
  }
}

struct CinmLoopPermutationPass
    : public impl::CinmLoopPermutationPassBase<CinmLoopPermutationPass> {
  using Base::Base;

  void runOnOperation() final {
    getOperation()->walk([](Operation *op) {
      SmallVector<affine::AffineForOp> loops;
      if (affine::AffineForOp outer = dyn_cast<affine::AffineForOp>(op)) {
        affine::getPerfectlyNestedLoops(loops, outer);
      }

      if (loops.size() >= 2) {
        SmallVector<unsigned> permMap;
        for (size_t i = 0; i < loops.size(); i++) {
          permMap.push_back(loops.size() - 1 - i);
        }

        if (isValidLoopInterchangePermutation(loops, permMap)) {
          cinm::permuteLoops(loops, permMap);
        }
      }
    });
  }
};

struct CinmLoopFusionPass
    : public impl::CinmLoopFusionPassBase<CinmLoopFusionPass> {
  using Base::Base;

  void runOnOperation() final {
    SmallVector<SmallVector<affine::AffineForOp>> fusableLoopSequences;

    getOperation()->walk([&](Operation *op) -> void {
      for (size_t i = 0; i < op->getNumRegions(); i++) {
        for (auto &block : op->getRegions()[i].getBlocks()) {
          auto iterator = block.begin();
          while (iterator != block.end()) {
            if (auto first = dyn_cast<affine::AffineForOp>(*iterator++)) {
              if (!first.hasConstantBounds()) {
                continue;
              }

              SmallVector<affine::AffineForOp> loopSequence{first};
              const int64_t lowerBound = first.getConstantLowerBound();
              const int64_t upperBound = first.getConstantUpperBound();
              const int64_t step = first.getStepAsInt();

              while (auto next = dyn_cast<affine::AffineForOp>(*iterator)) {
                if (!next.hasConstantBounds()) {
                  break;
                }

                if (next.getConstantLowerBound() != lowerBound ||
                    next.getConstantUpperBound() != upperBound ||
                    next.getStepAsInt() != step) {
                  break;
                }

                loopSequence.push_back(next);
                iterator++;
              }

              if (loopSequence.size() >= 2) {
                fusableLoopSequences.push_back(loopSequence);
              }
            }
          }
        }
      }
    });

    for (auto &loops : fusableLoopSequences) {
      fuseLoops(loops);
    }
  }
};

} // namespace mlir::cinm
