#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include <llvm/Support/Regex.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_UPMEMOUTLINEKERNELPASS
#include "cinm-mlir/Dialect/UPMEM/Transforms/Passes.h.inc"

/// Outline the `gpu.launch` operation body into a kernel function. Replace
/// `gpu.terminator` operations by `gpu.return` in the generated function.
/// Set block and grid size bounds if known.
static upmem::UPMEMFuncOp outlineKernelFuncImpl(upmem::LaunchOp launchOp,
                                                StringRef kernelFnName,
                                                SetVector<Value> &operands) {
  Location loc = launchOp.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation.
  OpBuilder builder(launchOp.getContext());
  Region &launchOpBody = launchOp.getBody();

  FunctionType type = FunctionType::get(launchOp.getContext(), {}, {});
  auto outlinedFunc =
      builder.create<upmem::UPMEMFuncOp>(loc, kernelFnName, type);

  outlinedFunc->setAttr(upmem::UPMEMDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());

  IRMapping map;
  Region &outlinedFuncBody = outlinedFunc.getBody();
  Block &outlinedEntryBlock = outlinedFuncBody.front();

  Block &launchOpEntry = launchOpBody.front();

  ///  CLone the region into the func, we remap the block arguments
  {
    auto taskletArg = launchOpEntry.getArgument(2);
    auto taskletId = builder.create<upmem::TaskletIDOp>(taskletArg.getLoc());
    map.map(taskletArg, taskletId);
    outlinedEntryBlock.push_back(taskletId);

    builder.setInsertionPointToEnd(&outlinedEntryBlock);

    for (auto &op : launchOpEntry.without_terminator()) {
      builder.clone(op, map);
    }
    builder.create<upmem::ReturnOp>(launchOpEntry.getTerminator()->getLoc());
  }

  return outlinedFunc;
}


static void convertToLaunchFuncOp(upmem::LaunchOp launchOp,
                                  upmem::UPMEMFuncOp kernelFunc,
                                  ValueRange operands) {
  OpBuilder builder(launchOp);
  // The launch op has an optional dynamic shared memory size. If it doesn't
  // exist, we use zero.
  Value asyncToken = launchOp.getAsyncToken();
  auto launchFunc = builder.create<upmem::LaunchFuncOp>(
      launchOp.getLoc(), kernelFunc, launchOp.getDeviceHierarchy(),
      launchOp.getDynamicSharedMemorySize(), operands,
      asyncToken ? asyncToken.getType() : nullptr,
      launchOp.getAsyncDependencies());
  launchOp.replaceAllUsesWith(launchFunc);
  launchOp.erase();
}


//===----------------------------------------------------------------------===//
struct UPMEMOutlineKernelPass
    : public impl::UPMEMOutlineKernelPassBase<UPMEMOutlineKernelPass> {
  using Base::Base;

  void runOnOperation() final;

  void getDependentDialects(DialectRegistry &registry) const override {}
  upmem::UPMEMModuleOp createKernelModule(upmem::UPMEMFuncOp kernelFunc,
                                          const SymbolTable &parentSymbolTable);
};

void UPMEMOutlineKernelPass::runOnOperation() {
  SymbolTable symbolTable(getOperation());
  bool modified = false;
  for (auto func : getOperation().getOps<func::FuncOp>()) {
    Block::iterator insertPt(func->getNextNode());
    auto funcWalkResult = func.walk([&](upmem::LaunchOp op) {
      SetVector<Value> operands;

      upmem::UPMEMFuncOp outlinedFunc =
          outlineKernelFuncImpl(op, "main", operands);

      auto kernelModule = createKernelModule(outlinedFunc, symbolTable);
      symbolTable.insert(kernelModule, insertPt);

      //     // Potentially changes signature, pulling in constants.
      convertToLaunchFuncOp(op, outlinedFunc, operands.getArrayRef());
      modified = true;
      return WalkResult::advance();
    });
    if (funcWalkResult.wasInterrupted())
      return signalPassFailure();
  }

  // // If any new module was inserted in this module, annotate this module as
  // // a container module.
  // if (modified)
  //   getOperation()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
  //                           UnitAttr::get(&getContext()));
}

upmem::UPMEMModuleOp UPMEMOutlineKernelPass::createKernelModule(
    upmem::UPMEMFuncOp kernelFunc, const SymbolTable &parentSymbolTable) {
  auto *context = getOperation().getContext();
  OpBuilder builder(context);
  auto kernelModule = builder.create<upmem::UPMEMModuleOp>(
      kernelFunc.getLoc(), kernelFunc.getName());

  SymbolTable symbolTable(kernelModule);
  symbolTable.insert(kernelFunc);

  SmallVector<Operation *, 8> symbolDefWorklist = {kernelFunc};
  while (!symbolDefWorklist.empty()) {
    if (std::optional<SymbolTable::UseRange> symbolUses =
            SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
      for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
        StringRef symbolName =
            cast<FlatSymbolRefAttr>(symbolUse.getSymbolRef()).getValue();
        if (symbolTable.lookup(symbolName))
          continue;

        Operation *symbolDefClone =
            parentSymbolTable.lookup(symbolName)->clone();
        symbolDefWorklist.push_back(symbolDefClone);
        symbolTable.insert(symbolDefClone);
      }
    }
  }

  return kernelModule;
}

} // namespace mlir
