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
#include <athena/backend/llvm/LLVMTrivialAllocator.h>
#include <athena/core/Graph.h>
#include <athena/core/InputNode.h>
#include <athena/core/Node.h>
#include <athena/core/Tensor.h>
#include <athena/ops/AddOperation.h>

using namespace athena::core;
using namespace athena::ops;
using namespace athena::backend::llvm;

bool testVectorSum() {
    // Arrange
    TensorShape vector({3});
    Tensor a(DataType::FLOAT, vector);
    Tensor b(DataType::FLOAT, vector);
    Tensor result(DataType::FLOAT, vector);

    InputNode* aInp = new InputNode(&a);
    InputNode* bInp = new InputNode(&b);

//    auto &&addOp = std::make_unique<AddOperation>();
    AddOperation addOp;
    Node* add = new Node(std::move(addOp));

    add->after(aInp);
    add->after(bInp);

    Graph graph(add);

    LLVMExecutor executor;
    std::unique_ptr<Allocator> trivialAllocator =
        std::make_unique<LLVMTrivialAllocator>();
    executor.setAllocator(trivialAllocator);
    executor.prepare(graph);

    // Act
    executor.execute();

    // Assert

//    auto pRes = (float*)executor.getAllocator()->getFastPointer(result);
//    if (abs(pRes[0] - 5.0) > 0.1) {
//        std::cout << "Element 0 is wrong"
//                  << "\n";
//        return 1;
//    }
//    if (abs(pRes[1] - 7.0) > 0.1) {
//        std::cout << "Element 1 is wrong"
//                  << "\n";
//        return 1;
//    }
//    if (abs(pRes[2] - 9.0) > 0.1) {
//        std::cout << "Element 2 is wrong"
//                  << "\n";
//        return 1;
//    }
    return true;
}

int main() {
    testVectorSum();
    return 0;
}
