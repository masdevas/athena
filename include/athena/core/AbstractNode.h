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


#ifndef ATHENA_ABSTRACTNODE_H
#define ATHENA_ABSTRACTNODE_H

#include <string>
#include <vector>

namespace athena::core {

class Graph;

enum class NodeType {
    INPUT,
    DEFAULT
};

class AbstractNode {
    friend class Graph;
 protected:
    std::vector<AbstractNode*> mOutgoingNodes;
    std::string mName;
    static size_t mNodeCounter;
    NodeType mNodeType;
    bool mWasVisitedFlag;

 public:
    AbstractNode() = delete;
    AbstractNode(const AbstractNode &node) = delete;
    AbstractNode(AbstractNode &&node) noexcept;
    explicit AbstractNode(std::string&& name, NodeType type);
    virtual ~AbstractNode() = default;
    AbstractNode& operator=(const AbstractNode& src) = delete;
    AbstractNode& operator=(AbstractNode&& src) noexcept;
    virtual void after(AbstractNode* node) = 0;
    void addOutgoingNode(AbstractNode* node);

    NodeType getType() {
        return mNodeType;
    }
};

}

#endif //ATHENA_ABSTRACTNODE_H
