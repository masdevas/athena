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

#include <athena/core/InputNode.h>
#include <athena/core/Node.h>
#include <athena/core/Traversal.h>
#include <athena/core/inner/Table.h>

#include <unordered_map>
#include <set>
#include <unordered_set>

namespace athena::core {
namespace inner {
struct Edge {
    size_t startNodeIndex;
    size_t endNodeIndex;
    EdgeMark mark;
    Edge(size_t startNodeIndex, size_t endNodeIndex, EdgeMark mark)
        : startNodeIndex(startNodeIndex),
          endNodeIndex(endNodeIndex),
          mark(mark) {}
    bool operator==(const Edge& rhs) const {
        return startNodeIndex == rhs.startNodeIndex &&
               endNodeIndex == rhs.endNodeIndex;
    }
    bool operator<(const Edge& rhs) const {
        return startNodeIndex < rhs.startNodeIndex;
    }
};
}

using SyncStorage = std::unordered_set<size_t>;
using OwningStorage = inner::TupleContainers<std::vector, Node, InputNode>::Holder;
using Topology = std::vector<inner::Edge>;

class Graph {
 private:
    SyncStorage mSyncStorage;
    OwningStorage mOwningStorage;
    Topology mTopology;
    size_t mGraphIndex;
    template <typename TemplateNodeType>
    void saveRealNode(TemplateNodeType& node, bool isRepairedNode, bool isErase);
    void saveNode(AbstractNode &node, bool isRepairedNode, bool isErase);
    template <typename TemplateNodeType>
    void initVisitsOf(std::unordered_map<size_t, inner::NodeState>& visits) const;
    void fullClear();

 public:
    Graph();
    Graph(const Graph& rhs) = delete;
    Graph(Graph&& rhs) noexcept;
    ~Graph();

    Graph &operator=(const Graph& rhs) = delete;
    Graph &operator=(Graph&& rhs) noexcept;

    const SyncStorage& getSyncStorage() const;
    const OwningStorage& getOwningStorage() const;
    const Topology& getTopology() const;
    void addNode(AbstractNode &node);
    void saveNode(AbstractNode &node, bool isRepairedNode = true);
    void saveAllSyncNodes(bool isRepairedNode = true);
    void removeNode(AbstractNode &node);
    void link(const AbstractNode &startNode, const AbstractNode &endNode, EdgeMark mark);
    size_t countOwningNodes() const;
    size_t countSyncNodes() const;
    size_t getGraphIndex() const;
    void clear();
    Traversal traverse(bool isRepairedNodes = true);
};
}  // namespace athena::core

#endif  // ATHENA_GRAPH_H
