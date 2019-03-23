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

#include <athena/backend/llvm/LLVMExecutor.h>
#include <athena/backend/llvm/LLVMGenerator.h>
#include <athena/core/InputNode.h>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/TargetSelect.h"

namespace athena::backend::llvm {

void LLVMExecutor::prepare(athena::core::Graph &graph) {
    mMainModule = std::make_unique<::llvm::Module>("AthenaMain",
                                                   mJITCompiler->getContext());
    mMainModule->setDataLayout(mJITCompiler->getDataLayout());

    ::llvm::FunctionType *FT = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(mJITCompiler->getContext()), false);
    ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage, "jitmain",
                             *mMainModule);

    LLVMGenerator generator(mJITCompiler->getContext(), mMainModule,
                            *mAllocator);

    auto [code, data] = graph.traverse();

    std::stack<core::Tensor *> preparedTensors;

    while (!code.empty()) {
        auto currentNode = code.front();

        if (currentNode->getType() == athena::core::NodeType::DEFAULT) {
            auto node = static_cast<core::Node *>(currentNode);
            auto &op  = node->getAssignedOperation();

            auto tensor = data.front();
            data.pop_front();
            preparedTensors.push(tensor);

            mVMAllocationTable.registerTensor(tensor);
            generator.generateAllocation(*tensor);

            op.gen(generator, preparedTensors);
            preparedTensors.empty();
        } else if (currentNode->getType() == core::NodeType::INPUT) {
            auto tensor = data.front();
            data.pop_front();

            mVMAllocationTable.registerTensor(tensor);
            generator.generateAllocation(*tensor);

            preparedTensors.push(tensor);
        }

        code.pop();
    }

    auto builder = generator.getBuilder();

    builder.CreateRetVoid();
}

void LLVMExecutor::execute() {
    auto err = mJITCompiler->addModule(mMainModule);
    if (err) {
        new core::FatalError("Error adding module to JIT");
    }

    auto sym = mJITCompiler->lookup("jitmain");
#ifdef DEBUG
    assert(sym && "Failed to find jitmain function");
#endif
    auto mainFunction = (void (*)())(intptr_t)sym.get().getAddress();
    mainFunction();
}

LLVMExecutor::LLVMExecutor() {
    ::llvm::InitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();

    auto compiler = AthenaJIT::create();

    if (!compiler) {
        ::llvm::consumeError(compiler.takeError());
        new core::FatalError("Unable to create JIT compiler");
    }

    mJITCompiler = std::move(compiler.get());
}

std::unique_ptr<core::Allocator> &LLVMExecutor::getAllocator() {
    return mAllocator;
}
void LLVMExecutor::setAllocator(std::unique_ptr<core::Allocator> &allocator) {
    mAllocator = std::move(allocator);
}

}  // namespace athena::backend::llvm