#include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.h"
#include <cinm-mlir/Dialect/Cnm/IR/CnmOps.h>
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <utility>

namespace mlir::cnm {

#define GEN_PASS_DEF_CNMMERGEWORKGROUPSPASS
#define GEN_PASS_DEF_CNMMERGELAUNCHOPSPASS
#include <cinm-mlir/Dialect/Cnm/Transforms/Passes.h.inc>

void mergeWorkgroups(
    func::FuncOp parentFunction, WorkgroupType wgType,
    const SmallVector<std::pair<cnm::WorkgroupOp, cnm::FreeWorkgroupOp>>
        &workgroups) {
  OpBuilder builder(parentFunction.getBody());
  cnm::WorkgroupOp newAllocOp =
      builder.create<cnm::WorkgroupOp>(parentFunction.getLoc(), wgType);

  parentFunction.walk([&](func::ReturnOp ret, WalkStage) -> void {
    builder.setInsertionPoint(ret);
    builder.create<cnm::FreeWorkgroupOp>(parentFunction.getLoc(), newAllocOp);
  });

  for (const auto &[alloc, free] : workgroups) {
    free->erase();
    alloc->replaceAllUsesWith(newAllocOp);
    alloc->erase();
  }
}

void mergeLaunchOps(SmallVector<cnm::LaunchOp> &launchOps) {
  OpBuilder builder(launchOps[0]->getContext());
  builder.setInsertionPoint(launchOps[0]);

  SmallVector<Value> inBuffers, outBuffers;
  for (cnm::LaunchOp op : launchOps) {
    inBuffers.append(op.getInputs().begin(), op.getInputs().end());
    outBuffers.append(op.getOutBuffers().begin(), op.getOutBuffers().end());
  }

  cnm::LaunchOp mergedOp = builder.create<cnm::LaunchOp>(
      launchOps[0]->getLoc(), launchOps[0].getWg(), inBuffers, outBuffers);
  Block &launchBlock = mergedOp.getBody().emplaceBlock();
  DenseMap<Value, BlockArgument> arguments;
  for (auto input : mergedOp.getParams()) {
    if (auto inputTy = input.getType().dyn_cast<cnm::BufferType>()) {
      auto mappedTy =
          MemRefType::get(inputTy.getShape(), inputTy.getElementType());
      arguments[input] = launchBlock.addArgument(mappedTy, input.getLoc());
    } else {
      arguments[input] =
          launchBlock.addArgument(input.getType(), input.getLoc());
    }
  }

  for (cnm::LaunchOp op : launchOps) {
    mergedOp.getBody().front().getOperations().splice(
        mergedOp.getBody().front().end(), op.getBody().front().getOperations());

    // erase terminator from old launch op
    mergedOp.getBody().front().back().erase();

    for (size_t i = 0; i < op.getParams().size(); i++) {
      op.getBody().getArgument(i).replaceAllUsesWith(
          arguments[op.getParams()[i]]);
    }

    op.erase();
  }

  builder.setInsertionPointToEnd(&mergedOp.getBody().back());
  builder.create<cnm::TerminatorOp>(mergedOp.getLoc());
}

struct CnmMergeWorkgroupsPass
    : public impl::CnmMergeWorkgroupsPassBase<CnmMergeWorkgroupsPass> {
  using Base::Base;

  void runOnOperation() final {
    getOperation()->walk([](func::FuncOp function, WalkStage) -> void {
      DenseMap<WorkgroupType,
               SmallVector<std::pair<cnm::WorkgroupOp, cnm::FreeWorkgroupOp>>>
          workGroups;
      function.walk([&workGroups](cnm::WorkgroupOp allocOp, WalkStage) -> void {
        for (auto &use : allocOp.getResult().getUses()) {
          if (const auto &freeOp =
                  dyn_cast<cnm::FreeWorkgroupOp>(use.getOwner())) {
            workGroups[allocOp.getType()].push_back({allocOp, freeOp});
            return;
          }
        }
      });

      for (const auto &[type, wgs] : workGroups) {
        mergeWorkgroups(function, type, wgs);
      }
    });
  }
};

struct CnmMergeLaunchOpsPass
    : public impl::CnmMergeLaunchOpsPassBase<CnmMergeLaunchOpsPass> {
  using Base::Base;

  void runOnOperation() final {
    SmallVector<SmallVector<cnm::LaunchOp>> launchOps;

    getOperation()->walk([&](Operation *op) -> void {
      for (size_t i = 0; i < op->getNumRegions(); i++) {
        for (auto &block : op->getRegions()[i].getBlocks()) {
          auto iterator = block.begin();
          while (iterator != block.end()) {
            if (auto first = dyn_cast<cnm::LaunchOp>(*iterator++)) {
              SmallVector<cnm::LaunchOp> sequence{first};
              while (auto next = dyn_cast<cnm::LaunchOp>(*iterator)) {
                if (next.getWg() != first.getWg()) {
                  break;
                }

                sequence.push_back(next);
                iterator++;
              }

              if (sequence.size() >= 2) {
                launchOps.push_back(sequence);
              }
            }
          }
        }
      }
    });

    for (auto &sequence : launchOps) {
      mergeLaunchOps(sequence);
    }
  }
};

} // namespace mlir::cnm
