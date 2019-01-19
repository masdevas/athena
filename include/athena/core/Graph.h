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

#ifndef ATHENA_GRAPH_H
#define ATHENA_GRAPH_H

#include <athena/core/Node.h>
#include <stack>
#include <queue>
#include <deque>
#include "Tensor.h"
#include "Node.h"

namespace athena::core {

class Graph {
 private:
    Node* outputNode;
    Node* lossFunctionNode;

 public:
    Graph() : outputNode(nullptr), lossFunctionNode(nullptr) {};
    Graph(const Graph&) = delete;
    Graph(Graph&& src) noexcept;
    Graph& operator=(const Graph&) = delete;
    Graph& operator=(Graph&& src) noexcept;
    ~Graph();

    std::tuple<std::queue<AbstractNode*>, std::deque<Tensor*> > traverse();

};
Operation &Node::getAssignedOperation() {
    return mOperation;
}

}

#endif //ATHENA_GRAPH_H
