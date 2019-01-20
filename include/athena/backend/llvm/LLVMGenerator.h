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
#ifndef ATHENA_LLVMGENERATOR_H
#define ATHENA_LLVMGENERATOR_H

#include <athena/core/AbstractGenerator.h>
#include <athena/core/Allocator.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>

namespace athena::backend::llvm {
class LLVMGenerator : public core::AbstractGenerator {
 private:
    std::shared_ptr<::llvm::Module> mModule;
    ::llvm::LLVMContext &mContext;
    ::llvm::IRBuilder<> mBuilder;

    core::Allocator &mAllocator;
 public:
    explicit LLVMGenerator(::llvm::LLVMContext &ctx,
                            std::shared_ptr<::llvm::Module> module,
                            core::Allocator &allocator);
    void generateAdd(core::Tensor &a, core::Tensor &b, core::Tensor &c) override;

};
}

#endif //ATHENA_LLVMGENERATOR_H
