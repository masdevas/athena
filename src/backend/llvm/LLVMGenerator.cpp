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

#include <athena/backend/llvm/LLVMGenerator.h>
#include <athena/core/FatalError.h>
#include "utils.h"

namespace athena::backend::llvm {

llvm::LLVMGenerator::LLVMGenerator(::llvm::LLVMContext &ctx,
                                    std::shared_ptr<::llvm::Module> module,
                                    core::Allocator &allocator)
                                    : mModule(std::move(module)),
                                    mContext(ctx),
                                    mBuilder(::llvm::IRBuilder(ctx)),
                                    mAllocator(allocator) {

}
void LLVMGenerator::generateAdd(core::Tensor &a, core::Tensor &b, core::Tensor &c) {

    // todo handle different data types

    ::llvm::Function *calledFunction = mModule->getFunction("fadd");

    if (!calledFunction)
        calledFunction = impl::create_fadd_decl(mContext, *mModule);

    if (!calledFunction)
        new core::FatalError("Unknown function referenced");

    // todo check arg count

    std::vector<::llvm::Value *> ArgsV;
    mBuilder.CreateCall(calledFunction, ArgsV, "faddtmp");
}
}