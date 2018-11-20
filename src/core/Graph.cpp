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

athena::core::Graph::Graph(athena::core::Graph &&src) noexcept {
    outputNode = src.outputNode;
    lossFunctionNode = src.lossFunctionNode;
    src.lossFunctionNode = nullptr;
    src.outputNode = nullptr;
}
athena::core::Graph &athena::core::Graph::operator=(athena::core::Graph &&src) noexcept {
    outputNode = src.outputNode;
    lossFunctionNode = src.lossFunctionNode;
    src.lossFunctionNode = nullptr;
    src.outputNode = nullptr;

    return *this;
}
athena::core::Graph::~Graph() {
    std::queue<AbstractNode*> nodes;

    if (lossFunctionNode != nullptr) {
        nodes.push(lossFunctionNode);
    } else {
        nodes.push(outputNode);
    }

    while (!nodes.empty()) {
        AbstractNode* curAbstractNode = nodes.front();
        nodes.pop();

        auto curNode = dynamic_cast<Node*>(curAbstractNode);
        if (curNode != nullptr && !curNode->mIncomingNodes.empty()) {
            for(auto node : curNode->mIncomingNodes) {
                nodes.push(node);
            }
        }

        delete curAbstractNode;
    }

}
