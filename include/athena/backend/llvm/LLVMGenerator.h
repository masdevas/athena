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

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace athena::backend::llvm {
class LLVMGenerator : public core::AbstractGenerator {
    private:
    const std::unique_ptr<::llvm::Module> &mModule;
    ::llvm::LLVMContext &mContext;
    // todo abatashev: refactor main block
    ::llvm::BasicBlock *mainBlock;
    ::llvm::IRBuilder<> mBuilder;

    core::Allocator &mAllocator;

    // AllocationTable mAllocationTable;
    ::llvm::Value *generateGetFastPointer(core::inner::Tensor &t);

    public:
    explicit LLVMGenerator(::llvm::LLVMContext &ctx,
                           const std::unique_ptr<::llvm::Module> &module,
                           core::Allocator &allocator);
    void generateAllocation(core::inner::Tensor &a) override;
    void generateAdd(core::inner::Tensor &a, core::inner::Tensor &b,
                     core::inner::Tensor &c) override;
    ::llvm::IRBuilder<> &getBuilder();
};
}  // namespace athena::backend::llvm

#endif  // ATHENA_LLVMGENERATOR_H
