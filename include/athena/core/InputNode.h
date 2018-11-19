/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#ifndef ATHENA_INPUTNODE_H
#define ATHENA_INPUTNODE_H

#include <athena/core/Node.h>
#include <athena/core/Tensor.h>

namespace athena::core {

class InputNode : public Node {
 private:
    Tensor& mTensor;

 public:
    InputNode() = delete;
    InputNode(const InputNode&) = delete;
    InputNode(InputNode&& src) noexcept;
    explicit InputNode(Tensor& tensor) : Node(Operation("nop")), mTensor(tensor) {};

    InputNode &operator=(const InputNode &) = delete;
    InputNode &operator=(InputNode&& src) noexcept;


    void after(Node* node) override {};
};

}

#endif //ATHENA_INPUTNODE_H
