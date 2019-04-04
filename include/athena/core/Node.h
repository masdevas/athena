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

#ifndef ATHENA_NODE_H
#define ATHENA_NODE_H

#include <athena/core/AbstractNode.h>
#include <athena/core/Operation.h>

namespace athena::core {
class Node : public AbstractNode {
 protected:
    const Operation *mOperation;
 public:
    Node() = delete;
    Node(const Node& rhs) = default;
    Node(Node&& rhs) noexcept = default;
    Node(TensorShape shape, DataType dataType, const Operation &operation, std::string name = "");
    ~Node() override;

    Node &operator=(const Node& rhs) = default;
    Node &operator=(Node&& rhs) noexcept = default;

    NodeType getType() const override;
    const Operation& getOperation() const;
    void clear() override;
};
}

#endif //ATHENA_NODE
