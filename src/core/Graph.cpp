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

#include <athena/core/Graph.h>
#include <athena/core/InputNode.h>
#include <athena/core/Tensor.h>

#include <queue>

namespace athena::core {
Graph::Graph(Graph &&src) noexcept {
    outputNode           = src.outputNode;
    lossFunctionNode     = src.lossFunctionNode;
    src.lossFunctionNode = nullptr;
    src.outputNode       = nullptr;
}
Graph &Graph::operator=(Graph &&src) noexcept {
    outputNode           = src.outputNode;
    lossFunctionNode     = src.lossFunctionNode;
    src.lossFunctionNode = nullptr;
    src.outputNode       = nullptr;

    return *this;
}
Graph::~Graph() {
    delete lossFunctionNode;
    std::queue<AbstractNode *> nodes;
    nodes.push(outputNode);
    while (nodes.size() > 0) {
        AbstractNode *got_node = nodes.front();
        nodes.pop();
        if (got_node->getType() == NodeType::DEFAULT) {
            Node *current_node = reinterpret_cast<Node *>(got_node);
            for (auto *node : current_node->mIncomingNodes) {
                nodes.push(node);
            }
            delete current_node;
        } else if (got_node->getType() == NodeType::INPUT) {
            delete reinterpret_cast<InputNode *>(got_node);
        } else {
            FatalError("Graph: Destructor: Type of node is not defined");
        }
    }
}
std::tuple<std::queue<AbstractNode *>, std::deque<Tensor *> >
Graph::traverse() {
    Node *startNode =
        lossFunctionNode == nullptr ? outputNode : lossFunctionNode;

    std::stack<AbstractNode *> dfsStack;
    dfsStack.push(startNode);
    std::deque<Tensor *> arguments;
    std::queue<AbstractNode *> graphQueue;

    while (!dfsStack.empty()) {
        AbstractNode *currentAbstractNode = dfsStack.top();

        if (currentAbstractNode->getType() == NodeType::DEFAULT) {
            auto currentNode  = static_cast<Node *>(currentAbstractNode);
            bool hasUnvisited = false;
            for (AbstractNode *node : currentNode->mIncomingNodes) {
                if (!node->mWasVisitedFlag) {
                    dfsStack.push(node);
                    hasUnvisited = true;
                }
            }
            if (!hasUnvisited) {
                graphQueue.push(currentAbstractNode);
                Tensor *result =
                    currentNode->getAssignedOperation().getResultSize(
                        arguments);
                arguments.push_back(result);
                dfsStack.pop();
            }
        } else {
            auto *inputNode = static_cast<InputNode *>(currentAbstractNode);
            arguments.push_back(inputNode->getData());
            graphQueue.push(currentAbstractNode);
            inputNode->mWasVisitedFlag = true;
            dfsStack.pop();
        }
    }

    return std::make_tuple(graphQueue, arguments);
}
}  // namespace athena::core