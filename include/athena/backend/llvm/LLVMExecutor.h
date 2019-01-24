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
#ifndef ATHENA_LLVMEXECUTOR_H
#define ATHENA_LLVMEXECUTOR_H

#include <athena/core/Executor.h>
#include <athena/core/Allocator.h>
#include <athena/backend/llvm/AthenaJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

namespace athena::backend::llvm {

class LLVMExecutor : public athena::core::Executor {
 private:
    AthenaJIT mJITCompiler;
    std::shared_ptr<::llvm::Module> mMainModule;
    std::unique_ptr<core::Allocator> mAllocator;
 public:
    void prepare(athena::core::Graph& graph) override;
    void execute() override {};
};

}

#endif //ATHENA_LLVMEXECUTOR_H
