//===- TranslateToCpp.cpp - Translating to C++ calls ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMDialect.h"
#include "cinm-mlir/Dialect/UPMEM/IR/UPMEMOps.h"
#include "cinm-mlir/Target/UPMEMCpp/UPMEMCppEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string>
#include <utility>

#define DEBUG_TYPE "translate-to-upmem-cpp"

using namespace mlir;
using namespace mlir::upmem_emitc;

using llvm::formatv;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

namespace {
/// Emitter that uses dialect specific emitters to emit C++ code.
struct CppEmitter {
  explicit CppEmitter(raw_ostream &os, bool declareVariablesAtTop);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  /// Emits an assignment for a variable which has been declared previously.
  LogicalResult emitVariableAssignment(OpResult result);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(OpResult result,
                                        bool trailingSemicolon);

  LogicalResult emitMemVariableDeclaration(OpResult result,
                                           bool trailingSemicolon);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits a label for the block.
  LogicalResult emitLabel(Block &block);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// Return the existing or a new label of a Block.
  StringRef getOrCreateName(Block &block);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(CppEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    CppEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Returns if all variables for op results and basic block arguments need to
  /// be declared at the beginning of a function.
  bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Boolean to enforce that all variables for op results and block
  /// arguments are declared at the beginning of the function. This also
  /// includes results from ops located in nested regions.
  bool declareVariablesAtTop;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
};
} // namespace

static LogicalResult printConstantOp(CppEmitter &emitter, Operation *operation,
                                     Attribute value) {
  OpResult result = operation->getResult(0);

  // Only emit an assignment as the variable was already declared when printing
  // the FuncOp.
  if (emitter.shouldDeclareVariablesAtTop()) {
    // Skip the assignment if the emitc.constant has no value.

    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    return emitter.emitAttribute(operation->getLoc(), value);
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}

static LogicalResult printValueOrConstant(CppEmitter &emitter, Value value) {
  Operation *op = value.getDefiningOp();
  if (arith::ConstantOp constant = dyn_cast_or_null<arith::ConstantOp>(op)) {
    if (emitter.emitAttribute(constant.getLoc(), constant.getValueAttr())
            .failed()) {
      return failure();
    }
  } else {
    emitter.ostream() << emitter.getOrCreateName(value);
  }
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    upmem::TaskletIDOp idOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*idOp)))
    return failure();
  os << "me()";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    upmem::BaseMRAMAddrOp heapOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*heapOp)))
    return failure();
  os << "(uint32_t) DPU_MRAM_HEAP_POINTER";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    upmem::PrivateWRAMAllocOp wramAllocOp) {
  raw_ostream &os = emitter.ostream();
  MemRefType res_type = dyn_cast<MemRefType>(wramAllocOp.getResult().getType());
  Type elementType = res_type.getElementType();

  os << "__dma_aligned ";
  if (emitter.emitType(wramAllocOp.getLoc(), elementType).failed()) {
    return failure();
  }

  size_t size = res_type.getNumElements();
  const size_t elementSize = elementType.getIntOrFloatBitWidth() / 8;
  if (size * elementSize < 8) {
    size = 8 / elementSize;
  }
  os << " " << emitter.getOrCreateName(wramAllocOp) << "[" << size << "]";

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    memref::LoadOp loadOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*loadOp)))
    return failure();

  os << emitter.getOrCreateName(loadOp->getOperand(0));
  if (loadOp->getNumOperands() > 1) {
    os << "[";
    if (printValueOrConstant(emitter, loadOp->getOperand(1)).failed()) {
      return failure();
    }
    os << "]";
  } else {
    os << "[0]";
  }
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    memref::StoreOp storeOp) {
  raw_ostream &os = emitter.ostream();
  os << emitter.getOrCreateName(storeOp->getOperand(1));
  if (storeOp.getNumOperands() > 2) {
    os << "[";
    if (printValueOrConstant(emitter, storeOp->getOperand(2)).failed()) {
      return failure();
    }
    os << "]";
  } else {
    os << "[0]";
  }
  os << " = ";

  if (printValueOrConstant(emitter, storeOp->getOperand(0)).failed()) {
    return failure();
  }
  return success();
}

static LogicalResult printMRAMCopy(CppEmitter &emitter, Location loc,
                                   upmem::MemcpyDirOp dir, Type elementType,
                                   Value from, Value to, size_t staticSize,
                                   Value dynamicSize, size_t offset) {
  raw_ostream &os = emitter.ostream();
  if (dir == upmem::MemcpyDirOp::MRAMToWRAM) {
    os << "mram_read((const __mram_ptr ";
  } else if (dir == upmem::MemcpyDirOp::WRAMToMRAM) {
    os << "mram_write((const ";
  }

  if (emitter.emitType(loc, elementType).failed()) {
    return failure();
  }

  os << "*)" << emitter.getOrCreateName(from);
  if (offset > 0) {
    os << " + " << offset;
  }
  os << ", ";

  if (dir == upmem::MemcpyDirOp::MRAMToWRAM) {
    os << "(";
  } else if (dir == upmem::MemcpyDirOp::WRAMToMRAM) {
    os << "(__mram_ptr ";
  }

  if (emitter.emitType(loc, elementType).failed()) {
    return failure();
  }

  os << "*)" << emitter.getOrCreateName(to);
  if (offset > 0) {
    os << " + " << offset;
  }
  os << ", ";

  if (dynamicSize) {
    if (printValueOrConstant(emitter, dynamicSize).failed()) {
      return failure();
    }
  } else {
    os << staticSize;
  }

  os << " * sizeof(";
  if (emitter.emitType(loc, elementType).failed()) {
    return failure();
  }
  os << "))";

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    upmem::MemcpyOp memcpyOp) {
  raw_ostream &os = emitter.ostream();
  auto direction = memcpyOp.getDirection();
  Value from, to;
  if (direction == upmem::MemcpyDirOp::MRAMToWRAM) {
    from = memcpyOp->getOperand(2);
    to = memcpyOp->getOperand(0);
  } else if (direction == upmem::MemcpyDirOp::WRAMToMRAM) {
    from = memcpyOp->getOperand(0);
    to = memcpyOp->getOperand(2);
  }

  Value size = memcpyOp->getOperand(1);
  Type elementType =
      dyn_cast<MemRefType>(memcpyOp.getOperand(0).getType()).getElementType();
  size_t elementSize = elementType.getIntOrFloatBitWidth() / 8;
  if (arith::ConstantOp staticSize =
          dyn_cast<arith::ConstantOp>(size.getDefiningOp())) {
    size_t remainingElements =
        llvm::dyn_cast<IntegerAttr>(staticSize.getValueAttr()).getInt();
    size_t offset = 0;
    while (remainingElements > 0) {
      size_t chunkSize = std::min(2048lu / elementSize, remainingElements);
      if (printMRAMCopy(emitter, memcpyOp.getLoc(), direction, elementType,
                        from, to, chunkSize, {}, offset)
              .failed()) {
        return failure();
      }
      offset += chunkSize;
      remainingElements -= chunkSize;
      if (remainingElements > 0) {
        os << ";\n";
      }
    }
  } else {
    if (printMRAMCopy(emitter, memcpyOp.getLoc(), direction, elementType, from,
                      to, 0, size, 0)
            .failed()) {
      return failure();
    }
  }

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    func::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValueAttr();

  return printConstantOp(emitter, operation, value);
}

static LogicalResult printBinaryOperation(CppEmitter &emitter,
                                          Operation *operation,
                                          StringRef binaryOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  if (printValueOrConstant(emitter, operation->getOperand(0)).failed()) {
    return failure();
  }
  os << " " << binaryOperator << " ";
  if (printValueOrConstant(emitter, operation->getOperand(1)).failed()) {
    return failure();
  }

  return success();
}

// arith ops

static LogicalResult printOperation(CppEmitter &emitter, arith::AddFOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "+");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::AddIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "+");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::AddUIExtendedOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "+");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::AndIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "&");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::BitcastOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::CeilDivSIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::CeilDivUIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::CmpFOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::CmpIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ConstantOp op) {
  return printConstantOp(emitter, op.getOperation(), op.getValue());
}

static LogicalResult printOperation(CppEmitter &emitter, arith::DivFOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "/");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::DivSIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "/");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::DivUIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "/");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ExtFOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ExtSIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ExtUIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::FloorDivSIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::FPToSIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::FPToUIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::IndexCastOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::IndexCastUIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MaximumFOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MaxNumFOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MaxSIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MaxUIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MinimumFOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MinNumFOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MinSIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MinUIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MulFOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "*");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MulIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "*");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::MulSIExtendedOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::MulUIExtendedOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::NegFOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::OrIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "|");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::RemFOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::RemSIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "%");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::RemUIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "%");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::SelectOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ShLIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "<<");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ShRSIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), ">>");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::ShRUIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), ">>");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::SIToFPOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::SubFOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "-");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::SubIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "-");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::TruncFOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::TruncIOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::UIToFPOp op) {
  assert(false && "todo: implement op printer");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::XOrIOp op) {
  return printBinaryOperation(emitter, op.getOperation(), "^");
}

static LogicalResult printOperation(CppEmitter &emitter, LLVM::ExpOp op) {
  if (emitter.emitAssignPrefix(*op.getOperation()).failed()) {
    return failure();
  }

  emitter.ostream() << "expf(" << emitter.getOrCreateName(op.getOperand())
                    << ")";

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    cf::BranchOp branchOp) {
  raw_ostream &os = emitter.ostream();
  Block &successor = *branchOp.getSuccessor();

  for (auto pair :
       llvm::zip(branchOp.getOperands(), successor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(successor)))
    return branchOp.emitOpError("unable to find label for successor block");
  os << emitter.getOrCreateName(successor);
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    cf::CondBranchOp condBranchOp) {
  raw_indented_ostream &os = emitter.ostream();
  Block &trueSuccessor = *condBranchOp.getTrueDest();
  Block &falseSuccessor = *condBranchOp.getFalseDest();

  os << "if (" << emitter.getOrCreateName(condBranchOp.getCondition())
     << ") {\n";

  os.indent();

  // If condition is true.
  for (auto pair : llvm::zip(condBranchOp.getTrueOperands(),
                             trueSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(trueSuccessor))) {
    return condBranchOp.emitOpError("unable to find label for successor block");
  }
  os << emitter.getOrCreateName(trueSuccessor) << ";\n";
  os.unindent() << "} else {\n";
  os.indent();
  // If condition is false.
  for (auto pair : llvm::zip(condBranchOp.getFalseOperands(),
                             falseSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(falseSuccessor))) {
    return condBranchOp.emitOpError()
           << "unable to find label for successor block";
  }
  os << emitter.getOrCreateName(falseSuccessor) << ";\n";
  os.unindent() << "}";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, func::CallOp callOp) {
  if (failed(emitter.emitAssignPrefix(*callOp.getOperation())))
    return failure();

  raw_ostream &os = emitter.ostream();
  os << callOp.getCallee() << "(";
  if (failed(emitter.emitOperands(*callOp.getOperation())))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printDPUReset(CppEmitter &emitter) {
  raw_ostream &os = emitter.ostream();

  os << "dpu_reset();\n";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, scf::ForOp forOp) {

  raw_indented_ostream &os = emitter.ostream();

  OperandRange operands = forOp.getInitArgs();
  Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
  Operation::result_range results = forOp.getResults();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : results) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  for (auto pair : llvm::zip(iterArgs, operands)) {
    if (failed(emitter.emitType(forOp.getLoc(), std::get<0>(pair).getType())))
      return failure();
    os << " " << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    os << emitter.getOrCreateName(std::get<1>(pair)) << ";";
    os << "\n";
  }

  os << "for (";
  if (failed(
          emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " = ";
  if (printValueOrConstant(emitter, forOp.getLowerBound()).failed()) {
    return failure();
  }
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  if (printValueOrConstant(emitter, forOp.getUpperBound()).failed()) {
    return failure();
  }
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  if (printValueOrConstant(emitter, forOp.getStep()).failed()) {
    return failure();
  }
  os << ") {\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op because this updates the result variables
  // of the for op in the generated code. Instead we update the iterArgs at
  // the end of a loop iteration and set the result variables after the for
  // loop.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
      return failure();
  }

  Operation *yieldOp = forRegion.getBlocks().front().getTerminator();
  // Copy yield operands into iterArgs at the end of a loop iteration.
  for (auto pair : llvm::zip(iterArgs, yieldOp->getOperands())) {
    BlockArgument iterArg = std::get<0>(pair);
    Value operand = std::get<1>(pair);
    os << emitter.getOrCreateName(iterArg) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os.unindent() << "}";

  // Copy iterArgs into results after the for loop.
  for (auto pair : llvm::zip(results, iterArgs)) {
    OpResult result = std::get<0>(pair);
    BlockArgument iterArg = std::get<1>(pair);
    os << "\n"
       << emitter.getOrCreateName(result) << " = "
       << emitter.getOrCreateName(iterArg) << ";";
  }

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, scf::IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : ifOp.getResults()) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  os << "if (";
  if (failed(emitter.emitOperands(*ifOp.getOperation())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &thenRegion = ifOp.getThenRegion();
  for (Operation &op : thenRegion.getOps()) {
    // Note: This prints a superfluous semicolon if the terminating yield op has
    // zero results.
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
      return failure();
  }

  os.unindent() << "}";

  Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    os << " else {\n";
    os.indent();

    for (Operation &op : elseRegion.getOps()) {
      // Note: This prints a superfluous semicolon if the terminating yield op
      // has zero results.
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
        return failure();
    }

    os.unindent() << "}";
  }

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, scf::YieldOp yieldOp) {
  raw_ostream &os = emitter.ostream();
  Operation &parentOp = *yieldOp.getOperation()->getParentOp();

  if (yieldOp.getNumOperands() != parentOp.getNumResults()) {
    return yieldOp.emitError("number of operands does not to match the number "
                             "of the parent op's results");
  }

  if (failed(interleaveWithError(
          llvm::zip(parentOp.getResults(), yieldOp.getOperands()),
          [&](auto pair) -> LogicalResult {
            auto result = std::get<0>(pair);
            auto operand = std::get<1>(pair);
            os << emitter.getOrCreateName(result) << " = ";

            if (!emitter.hasValueInScope(operand))
              return yieldOp.emitError("operand value not in scope");
            os << emitter.getOrCreateName(operand);
            return success();
          },
          [&]() { os << ";\n"; })))
    return failure();

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << " " << emitter.getOrCreateName(returnOp.getOperand(0));
    return success(emitter.hasValueInScope(returnOp.getOperand(0)));
  default:
    os << " std::make_tuple(";
    if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
      return failure();
    os << ")";
    return success();
  }
}

static LogicalResult printOperation(CppEmitter &emitter, ModuleOp moduleOp) {
  CppEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    upmem::UPMEMFuncOp functionOp) {
  // We need to declare variables at top if the function has multiple blocks.
  if (!emitter.shouldDeclareVariablesAtTop() &&
      functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }

  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  // if (failed(emitter.emitTypes(functionOp.getLoc(),
  //  functionOp.getFunctionType().getResults())))
  // return failure();
  os << "void " << functionOp.getName();

  os << "(";
  if (failed(interleaveCommaWithError(
          functionOp.getArguments(), os,
          [&](BlockArgument arg) -> LogicalResult {
            if (failed(emitter.emitType(functionOp.getLoc(), arg.getType())))
              return failure();
            os << " " << emitter.getOrCreateName(arg);
            return success();
          })))
    return failure();
  os << ") {\n";
  os.indent();
  if (emitter.shouldDeclareVariablesAtTop()) {
    // Declare all variables that hold op results including those from nested
    // regions.
    WalkResult result =
        functionOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          for (OpResult result : op->getResults()) {
            if (failed(emitter.emitVariableDeclaration(
                    result, /*trailingSemicolon=*/true))) {
              return WalkResult(
                  op->emitError("unable to declare result variable for op"));
            }
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return failure();
  }

  Region::BlockListType &blocks = functionOp.getBlocks();
  // Create label names for basic blocks.
  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Declare variables for basic block arguments.
  for (Block &block : llvm::drop_begin(blocks)) {
    for (BlockArgument &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg))
        return functionOp.emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      if (failed(
              emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (Block &block : blocks) {
    // Only print a label if the block has predecessors.
    if (!block.hasNoPredecessors()) {
      if (failed(emitter.emitLabel(block)))
        return failure();
    }
    for (Operation &op : block.getOperations()) {
      // When generating code for an scf.if or cf.cond_br op no semicolon needs
      // to be printed after the closing brace.
      // When generating code for an scf.for op, printing a trailing semicolon
      // is handled within the printOperation function.
      bool trailingSemicolon =
          !isa<cf::CondBranchOp, scf::IfOp, scf::ForOp>(op);

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }
  os.unindent() << "}\n";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    upmem::ReturnOp returnOp) {
  emitter.ostream() << "return";
  return success();
}

static void printCompilationVar(upmem::UPMEMFuncOp &kernel, raw_ostream &os) {
  os << "COMPILE_" << kernel.getSymName();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    upmem::UPMEMModuleOp moduleOp) {
  CppEmitter::Scope scope(emitter);

  llvm::SmallVector<upmem::UPMEMFuncOp> kernels;
  for (Operation &op : moduleOp) {
    if (llvm::isa<upmem::UPMEMFuncOp>(op)) {
      kernels.push_back(llvm::cast<upmem::UPMEMFuncOp>(op));
    }
  }

  if (kernels.empty())
    return failure();

  raw_ostream &os = emitter.ostream();

  os << "// UPMEM-TRANSLATE: ";
  for (auto kernel : kernels) {
    printCompilationVar(kernel, os);
    os << ":" << kernel.getNumTasklets();
    os << ":" << kernel.getSymName(); // name of the binary
    os << ";";
  }

  os << "\n\n";

  os << "#include <alloc.h>\n"
        "#include <barrier.h>\n"
        "#include <defs.h>\n"
        "#include <mram.h>\n"
        "#include <perfcounter.h>\n\n"
        "#include <stdint.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n\n"
        "#include \"expf.c\"\n"
        "\n";

  for (auto kernel : kernels) {
    os << "#ifdef ";
    printCompilationVar(kernel, os);
    os << "\n";
    if (failed(printOperation(emitter, kernel)))
      return failure();
    os << "#endif\n\n";
  }

  os << "BARRIER_INIT(my_barrier, NR_TASKLETS);\n\n";

  os << "int main(void) {\n";
  os << "  barrier_wait(&my_barrier);\n";
  os << "  mem_reset();\n";
  for (auto kernel : kernels) {
    os << "#ifdef ";
    printCompilationVar(kernel, os);
    os << "\n";
    os << "  " << kernel.getName() << "();\n";
    os << "#endif\n";
  }
  os << "  mem_reset();\n";
  os << "  return 0;\n";
  os << "}";

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    func::FuncOp functionOp) {
  return success();
}

CppEmitter::CppEmitter(raw_ostream &os, bool declareVariablesAtTop)
    : os(os), declareVariablesAtTop(declareVariablesAtTop) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

/// Return the existing or a new name for a Value.
StringRef CppEmitter::getOrCreateName(Value val) {
  if (!valueMapper.count(val))
    valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *valueMapper.begin(val);
}

/// Return the existing or a new label for a Block.
StringRef CppEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}

bool CppEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

bool CppEmitter::hasValueInScope(Value val) { return valueMapper.count(val); }

bool CppEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}

LogicalResult CppEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
      case llvm::APFloatBase::S_IEEEsingle:
        os << "(float)";
        break;
      case llvm::APFloatBase::S_IEEEdouble:
        os << "(double)";
        break;
      default:
        break;
      };
      os << strValue;
    } else if (val.isNaN()) {
      os << "NAN";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        os << "-";
      os << "INFINITY";
    }
  };

  // Print floating point attributes.
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    printFloat(fAttr.getValue());
    return success();
  }
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
    os << '{';
    interleaveComma(dense, os, [&](const APFloat &val) { printFloat(val); });
    os << '}';
    return success();
  }

  // Print integer attributes.
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os, [&](const APInt &val) {
        printInt(val, shouldMapToUnsigned(iType.getSignedness()));
      });
      os << '}';
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os,
                      [&](const APInt &val) { printInt(val, false); });
      os << '}';
      return success();
    }
  }

  // Print symbolic reference attributes.
  if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
    if (sAttr.getNestedReferences().size() > 1)
      return emitError(loc, "attribute has more than 1 nested reference");
    os << sAttr.getRootReference().getValue();
    return success();
  }

  // Print type attributes.
  if (auto type = dyn_cast<TypeAttr>(attr))
    return emitType(loc, type.getValue());

  return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult CppEmitter::emitOperands(Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    if (!hasValueInScope(result))
      return op.emitOpError() << "operand value not in scope";
    os << getOrCreateName(result);
    return success();
  };
  return interleaveCommaWithError(op.getOperands(), os, emitOperandName);
}

LogicalResult
CppEmitter::emitOperandsAndAttributes(Operation &op,
                                      ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op)))
    return failure();
  // Insert comma in between operands and non-filtered attributes if needed.
  if (op.getNumOperands() > 0) {
    for (NamedAttribute attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.getName().strref())) {
        os << ", ";
        break;
      }
    }
  }
  // Emit attributes.
  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.getName().strref()))
      return success();
    os << "/* " << attr.getName().getValue() << " */";
    if (failed(emitAttribute(op.getLoc(), attr.getValue())))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult CppEmitter::emitVariableAssignment(OpResult result) {
  if (!hasValueInScope(result)) {
    return result.getDefiningOp()->emitOpError(
        "result variable for the operation has not been declared");
  }
  os << getOrCreateName(result) << " = ";
  return success();
}

LogicalResult CppEmitter::emitVariableDeclaration(OpResult result,
                                                  bool trailingSemicolon) {
  if (hasValueInScope(result)) {
    return result.getDefiningOp()->emitError(
        "result variable for the operation already declared");
  }
  if (failed(emitType(result.getOwner()->getLoc(), result.getType())))
    return failure();
  os << " " << getOrCreateName(result);
  if (trailingSemicolon)
    os << ";\n";
  return success();
}

LogicalResult CppEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (shouldDeclareVariablesAtTop()) {
      if (failed(emitVariableAssignment(result)))
        return failure();
    } else {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
        return failure();
      os << " = ";
    }
    break;
  }
  default:
    if (!shouldDeclareVariablesAtTop()) {
      for (OpResult result : op.getResults()) {
        if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
          return failure();
      }
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult CppEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block))
    return block.getParentOp()->emitError("label for block not found");
  // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
  // label instead of using `getOStream`.
  os.getOStream() << getOrCreateName(block) << ":\n";
  return success();
}

LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  if (dyn_cast<arith::ConstantOp>(op)) {
    return success();
  }

  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<upmem::UPMEMModuleOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // CF ops.
          .Case<cf::BranchOp, cf::CondBranchOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Arith ops
          .Case<arith::MulIOp, arith::AddIOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Func ops.
          .Case<func::CallOp, func::ConstantOp, func::FuncOp,
                upmem::UPMEMFuncOp, func::ReturnOp, upmem::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          // SCF ops.
          .Case<scf::ForOp, scf::IfOp, scf::YieldOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Arithmetic ops.
          .Case<arith::AddFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::AddIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::AddUIExtendedOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::AndIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::BitcastOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::CeilDivSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::CeilDivUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::CmpFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::CmpIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ConstantOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::DivFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::DivSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::DivUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ExtFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ExtSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ExtUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::FloorDivSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::FPToSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::FPToUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::IndexCastOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::IndexCastUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MaximumFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MaxNumFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MaxSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MaxUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MinimumFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MinNumFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MinSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MinUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MulFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MulIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MulSIExtendedOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MulUIExtendedOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::NegFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::OrIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::RemFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::RemSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::RemUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::SelectOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ShLIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ShRSIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::ShRUIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::SIToFPOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::SubFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::SubIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::TruncFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::TruncIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::UIToFPOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::XOrIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<LLVM::ExpOp>([&](auto op) { return printOperation(*this, op); })
          .Case<upmem::TaskletIDOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<upmem::BaseMRAMAddrOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<upmem::PrivateWRAMAllocOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<upmem::MemcpyOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<memref::LoadOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<memref::StoreOp>(
              [&](auto op) { return printOperation(*this, op); })
          // [&](auto op) { skipSemicolon = true; return success(); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();

  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult CppEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "uint" << iType.getWidth() << "_t"), success();
      else
        return (os << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto iType = dyn_cast<IndexType>(type))
    return (os << "int32_t"), success();
  if (auto tType = dyn_cast<TensorType>(type)) {
    if (!tType.hasRank())
      return emitError(loc, "cannot emit unranked tensor type");
    if (!tType.hasStaticShape())
      return emitError(loc, "cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (failed(emitType(loc, tType.getElementType())))
      return failure();
    auto shape = tType.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  }
  if (auto tType = dyn_cast<TupleType>(type))
    return emitTupleType(loc, tType.getTypes());
  if (auto pType = dyn_cast<MemRefType>(type)) {
    Type type = pType.getElementType();
    if (auto t = dyn_cast<IntegerType>(type)) {
      os << "int ";
    } else if (auto t = dyn_cast<FloatType>(type)) {
      os << "float ";
    }
    os << "*";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type;
}

LogicalResult CppEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return emitTupleType(loc, types);
  }
}

LogicalResult CppEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ">";
  return success();
}

LogicalResult upmem_emitc::UPMEMtranslateToCpp(Operation *op, raw_ostream &os,
                                               bool declareVariablesAtTop) {
  CppEmitter emitter(os, declareVariablesAtTop);
  LogicalResult res = success();
  op->walk<WalkOrder::PreOrder>([&](Operation *child) {
    if (llvm::isa<upmem::UPMEMModuleOp>(child)) {
      res = emitter.emitOperation(*child, /*trailingSemicolon=*/false);
      return WalkResult::interrupt();
    } else if (llvm::isa<ModuleOp>(child))
      return WalkResult::advance();
    return WalkResult::skip();
  });
  return res;
}
