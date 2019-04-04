/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://athenaframework.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "utils.h"

#include <athena/backend/llvm/LLVMGenerator.h>
#include <athena/core/FatalError.h>

namespace athena::backend::llvm {

llvm::LLVMGenerator::LLVMGenerator(
    ::llvm::LLVMContext &ctx, const std::unique_ptr<::llvm::Module> &module,
    core::Allocator &allocator)
    : mModule(module),
      mainBlock(::llvm::BasicBlock::Create(ctx, "entry",
                                           module->getFunction("jitmain"))),
      mContext(ctx),
      mBuilder(::llvm::IRBuilder(mainBlock)),
      mAllocator(allocator) {
    mBuilder.SetInsertPoint(mainBlock);
}
void LLVMGenerator::generateAdd(core::inner::Tensor &a, core::inner::Tensor &b,
                                core::inner::Tensor &c) {
    // todo handle different data types

    ::llvm::Function *calledFunction = mModule->getFunction("fadd");

    if (!calledFunction)
        calledFunction = impl::create_fadd_decl(mContext, *mModule);

    if (!calledFunction) {
        core::FatalError(1, "Unknown function referenced");
    }

    // todo check arg count

    std::vector<::llvm::Value *> ArgsV;

    ArgsV.push_back(generateGetFastPointer(a));
    ::llvm::Constant *aSizeConst = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(mContext), a.getShapeView().getTotalSize());
    ArgsV.push_back(aSizeConst);
    ArgsV.push_back(generateGetFastPointer(b));
    ::llvm::Constant *bSizeConst = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(mContext), b.getShapeView().getTotalSize());
    ArgsV.push_back(bSizeConst);
    ArgsV.push_back(generateGetFastPointer(c));
    mBuilder.CreateCall(calledFunction, ArgsV);
}

void LLVMGenerator::generateAllocation(core::inner::Tensor &a) {
    // todo handle different data types

    ::llvm::Function *calledFunction = mModule->getFunction("allocate");

    if (!calledFunction)
        calledFunction = impl::create_allocate_decl(mContext, *mModule);

    if (!calledFunction) {
        core::FatalError(1, "Unknown function referenced");
    }

    // todo check arg count

    std::vector<::llvm::Value *> ArgsV;
    ::llvm::Constant *allocatorConst = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(mContext), (size_t)(&mAllocator));
    ArgsV.push_back(allocatorConst);
    ::llvm::Constant *tensorConst = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(mContext), (size_t)(&a));
    ArgsV.push_back(tensorConst);
    mBuilder.CreateCall(calledFunction, ArgsV);
}

::llvm::IRBuilder<> &LLVMGenerator::getBuilder() { return mBuilder; }

::llvm::Value *LLVMGenerator::generateGetFastPointer(core::inner::Tensor &t) {
    ::llvm::Function *calledFunction = mModule->getFunction("get_fast_pointer");

    if (!calledFunction)
        calledFunction = impl::create_get_fast_pointer_decl(mContext, *mModule);

    if (!calledFunction) {
        core::FatalError(1, "Unknown function referenced");
    }

    std::vector<::llvm::Value *> ArgsV;
    ::llvm::Constant *allocatorConst = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(mContext), (size_t)(&mAllocator));
    ArgsV.push_back(allocatorConst);
    ::llvm::Constant *tensorConst = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(mContext), (size_t)(&t));
    ArgsV.push_back(tensorConst);
    auto callInst = mBuilder.CreateCall(calledFunction, ArgsV);
    return callInst;
}
}  // namespace athena::backend::llvm