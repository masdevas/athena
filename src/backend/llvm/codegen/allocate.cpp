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

namespace athena::backend::llvm::codegen {

::llvm::Function *create_allocate_decl(::llvm::LLVMContext &ctx,
                                       ::llvm::Module &module) {
    std::vector<::llvm::Type *> args(2, ::llvm::Type::getInt64Ty(ctx));
    ::llvm::FunctionType *FT =
        ::llvm::FunctionType::get(::llvm::Type::getVoidTy(ctx), args, false);

    ::llvm::Function *F = ::llvm::Function::Create(
        FT, ::llvm::Function::ExternalLinkage, "athena_allocate", &module);

    return F;
}

void registerAllocate(LLVMGenerator *generator) {
    std::function<void(::llvm::LLVMContext &, ::llvm::Module &,
                       ::llvm::IRBuilder<> &, core::inner::Tensor &)>
        f = [generator](::llvm::LLVMContext &ctx, ::llvm::Module &module,
                        ::llvm::IRBuilder<> &builder, core::inner::Tensor &a) {
            // todo handle different data types

            ::llvm::Function *calledFunction =
                module.getFunction("athena_allocate");

            if (!calledFunction)
                calledFunction = create_allocate_decl(ctx, module);

            if (!calledFunction) {
                core::FatalError(1, "Unknown function referenced");
            }

            // todo check arg count

            std::vector<::llvm::Value *> ArgsV;
            ::llvm::Constant *allocatorConst =
                ::llvm::ConstantInt::get(::llvm::Type::getInt64Ty(ctx),
                                         (size_t)(&generator->getAllocator()));
            ArgsV.push_back(allocatorConst);
            ::llvm::Constant *tensorConst = ::llvm::ConstantInt::get(
                ::llvm::Type::getInt64Ty(ctx), (size_t)(&a));
            ArgsV.push_back(tensorConst);
            builder.CreateCall(calledFunction, ArgsV);
        };

    generator->registerFunctor("allocate", f);
}
}  // namespace athena::backend::llvm::codegen