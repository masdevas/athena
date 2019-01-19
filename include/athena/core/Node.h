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

#include <vector>
#include <athena/core/Operation.h>
#include <athena/core/AbstractNode.h>

namespace athena::core {

class Graph;

/**
 * The class represents a single node of a computation graph. It has a name, a set of incoming nodes,
 * and a set of outgoing nodes. It also encapsulates Operation.
 */
class Node : public AbstractNode {
    friend class Graph;
 protected:
    std::vector<AbstractNode*> mIncomingNodes;
    Operation mOperation;

 public:
    Node() = delete;
    Node(const Node& src) = delete;
    Node(Node&& src) noexcept;
    explicit Node(Operation&& op);
    ~Node() override = default;
    Node& operator=(const Node& src) = delete;
    Node& operator=(Node&& src) noexcept;
    void after(AbstractNode* node) override;
    Operation& getAssignedOperation();
};

}

#endif //ATHENA_NODE_H
