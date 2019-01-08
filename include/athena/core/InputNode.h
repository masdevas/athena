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

#include "Tensor.h"
#include "AbstractNode.h"

namespace athena::core {

class InputNode : public AbstractNode {
 private:
    Tensor *mTensor;

 public:
    InputNode() = delete;
    InputNode(const InputNode &node) = delete;
    InputNode(InputNode &&node) noexcept;
    explicit InputNode(Tensor *tensor);
    ~InputNode() override = default;
    InputNode& operator=(const InputNode& src) = delete;
    InputNode& operator=(InputNode&& src) noexcept;
    void after(AbstractNode* node) override;
    Tensor* getData();
    template <class Generator, typename ...Args>
    void gen(Generator g, Tensor* tensor, Args... args) {};
};

}
#endif //ATHENA_INPUTNODE_H
