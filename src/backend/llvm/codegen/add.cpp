/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "common.h"

#include <athena/backend/llvm/LLVMGenerator.h>

namespace athena::backend::llvm::codegen {

::llvm::Function *create_fadd_decl(::llvm::LLVMContext &ctx,
                                   ::llvm::Module &module) {
    std::vector<::llvm::Type *> args(5, ::llvm::Type::getInt64Ty(ctx));
    ::llvm::FunctionType *FT =
        ::llvm::FunctionType::get(::llvm::Type::getVoidTy(ctx), args, false);

    ::llvm::Function *F = ::llvm::Function::Create(
        FT, ::llvm::Function::ExternalLinkage, "athena_fadd", &module);

    return F;
}

void registerAdd(LLVMGenerator *generator) {
    std::function<void(::llvm::LLVMContext &, ::llvm::Module &,
                       ::llvm::IRBuilder<> &, core::inner::Tensor &,
                       core::inner::Tensor &, core::inner::Tensor &)>
        f = [generator](::llvm::LLVMContext &ctx, ::llvm::Module &module,
                        ::llvm::IRBuilder<> &builder, core::inner::Tensor &a,
                        core::inner::Tensor &b, core::inner::Tensor &c) {
            // todo handle different data types

            ::llvm::Function *calledFunction =
                module.getFunction("athena_fadd");

            if (!calledFunction) calledFunction = create_fadd_decl(ctx, module);

            if (!calledFunction) {
                core::FatalError(1, "Unknown function referenced");
            }

            // todo check arg count

            std::vector<::llvm::Value *> ArgsV;

            ArgsV.push_back(generator->generateGetFastPointer(a));
            ::llvm::Constant *aSizeConst = ::llvm::ConstantInt::get(
                ::llvm::Type::getInt64Ty(ctx), a.getShapeView().getTotalSize());
            ArgsV.push_back(aSizeConst);
            ArgsV.push_back(generator->generateGetFastPointer(b));
            ::llvm::Constant *bSizeConst = ::llvm::ConstantInt::get(
                ::llvm::Type::getInt64Ty(ctx), b.getShapeView().getTotalSize());
            ArgsV.push_back(bSizeConst);
            ArgsV.push_back(generator->generateGetFastPointer(c));
            builder.CreateCall(calledFunction, ArgsV);
        };

    generator->registerFunctor("add", f);
}
}  // namespace athena::backend::llvm::codegen
