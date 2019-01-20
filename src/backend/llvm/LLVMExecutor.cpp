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
#include <athena/core/InputNode.h>
#include <athena/backend/llvm/LLVMGenerator.h>

namespace athena::backend::llvm {

void LLVMExecutor::prepare(athena::core::Graph &graph) {
    mMainModule = std::make_shared<::llvm::Module>("AthenaMain", mLLVMContext);

    // todo consider initializing some optimizers

    // todo Consider passing parameters with generator and dropping
    LLVMGenerator generator(mLLVMContext, mMainModule, *mAllocator);

    auto[code, data] = graph.traverse();

    std::stack<core::Tensor*> preparedTensors;

    while (!code.empty()) {
        auto currentNode = code.front();

        if (currentNode->getType() == athena::core::NodeType::DEFAULT) {
            auto node = static_cast<core::Node *>(currentNode);
            auto op = node->getAssignedOperation();

            auto tensor = data.front();
            data.pop_front();
            preparedTensors.push(tensor);
            // todo Result tensor also needs to be allocated. Consider removing InputNode::gen and calling generator directly

            op.gen(generator, preparedTensors, mMainModule);
        } else if (currentNode->getType() == core::NodeType::INPUT) {
            auto node = static_cast<core::InputNode *>(currentNode);

            auto tensor = data.front();
            data.pop_front();

            node->gen(generator, tensor);

            preparedTensors.push(tensor);
        }

        code.pop();
    }
}

}