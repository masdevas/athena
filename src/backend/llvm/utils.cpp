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

namespace athena::backend::llvm::impl {

::llvm::Function *create_fadd_decl(::llvm::LLVMContext &ctx, ::llvm::Module &module) {
    std::vector<::llvm::Type*> args(5);
    ::llvm::FunctionType *FT =
        ::llvm::FunctionType::get(::llvm::Type::getVoidTy(ctx), args, false);

    ::llvm::Function *F =
        ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage, "fadd", &module);

    return F;
}

}