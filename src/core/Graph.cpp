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
#include <queue>
#include <deque>
#include <athena/core/InputNode.h>
#include <tuple>
#include <athena/core/Tensor.h>

namespace athena::core {
Graph::Graph(Graph &&src) noexcept {
    outputNode = src.outputNode;
    lossFunctionNode = src.lossFunctionNode;
    src.lossFunctionNode = nullptr;
    src.outputNode = nullptr;
}
Graph &Graph::operator=(Graph &&src) noexcept {
    outputNode = src.outputNode;
    lossFunctionNode = src.lossFunctionNode;
    src.lossFunctionNode = nullptr;
    src.outputNode = nullptr;

    return *this;
}
Graph::~Graph() {
    std::queue<AbstractNode *> nodes;

    if (lossFunctionNode != nullptr) {
        nodes.push(lossFunctionNode);
    } else {
        nodes.push(outputNode);
    }

    while (!nodes.empty()) {
        AbstractNode *curAbstractNode = nodes.front();
        nodes.pop();

        auto curNode = static_cast<Node*>(curAbstractNode);
        if (curNode != nullptr && !curNode->mIncomingNodes.empty()) {
            for(auto node : curNode->mIncomingNodes) {
                nodes.push(node);
            }
        }

        delete curAbstractNode;
    }

}
std::tuple<std::queue<AbstractNode *>, std::deque<Tensor *> > Graph::traverse() {
    Node *startNode = lossFunctionNode == nullptr ? outputNode : lossFunctionNode;

    std::stack<AbstractNode *> dfsStack;
    dfsStack.push(startNode);
    std::deque<Tensor *> arguments;
    std::queue<AbstractNode *> graphQueue;

    while (!dfsStack.empty()) {
        AbstractNode *currentAbstractNode = dfsStack.top();

        if (currentAbstractNode->getType() == NodeType::DEFAULT) {
            auto currentNode = static_cast<Node *>(currentAbstractNode);
            bool hasUnvisited = false;
            for (AbstractNode *node : currentNode->mIncomingNodes) {
                if (!node->mWasVisitedFlag) {
                    dfsStack.push(node);
                    hasUnvisited = true;
                }
            }
            if (!hasUnvisited) {
                graphQueue.push(currentAbstractNode);
                Tensor *result = currentNode->mOperation.getResultSize(arguments);
                arguments.push_back(result);
                dfsStack.pop();
            }
        } else {
            auto *inputNode = static_cast<InputNode *>(currentAbstractNode);
            arguments.push_back(inputNode->getData());
            graphQueue.push(currentAbstractNode);
        }
    }

    return std::make_tuple(graphQueue, arguments);
}
}