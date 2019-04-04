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
#include <athena/core/inner/IndexFunctions.h>
#include <athena/core/inner/GlobalTables.h>

#include <algorithm>
#include <queue>

namespace athena::core {
Graph::Graph() : mGraphIndex(inner::getGraphTable().registerRecord(this)) {
}
Graph::Graph(Graph&& rhs) noexcept : mSyncStorage(std::move(rhs.mSyncStorage)),
                                     mOwningStorage(std::move(rhs.mOwningStorage)),
                                     mTopology(std::move(rhs.mTopology)),
                                     mGraphIndex(rhs.mGraphIndex) {
    inner::getGraphTable()[mGraphIndex] = this;
    rhs.fullClear();
}
Graph::~Graph() {
    inner::getGraphTable()[mGraphIndex] = nullptr;
    for (auto indexNode : mSyncStorage) {
        if (auto* node = inner::getNodeTable()[indexNode]; node) {
            inner::setGraphIndex(*(node), inner::kKUndefinedIndex);
        }
    }
}
Graph &Graph::operator=(Graph&& rhs) noexcept {
    mSyncStorage = std::move(rhs.mSyncStorage);
    mOwningStorage = std::move(rhs.mOwningStorage);
    mTopology = std::move(rhs.mTopology);
    mGraphIndex = rhs.mGraphIndex;
    inner::getGraphTable()[mGraphIndex] = this;
    rhs.fullClear();
    return *this;
}
template <typename TemplateNodeType>
void Graph::saveRealNode(TemplateNodeType& node, bool isRepairedNode, bool isErase) {
    if (isErase) {
        mSyncStorage.erase(node.getNodeIndex());
    }
    std::get<std::vector<TemplateNodeType>>(mOwningStorage).emplace_back(std::move(node));
    inner::setGraphIndex(node, inner::kKUndefinedIndex);
    if (isRepairedNode) {
        TemplateNodeType newNode(std::get<std::vector<TemplateNodeType>>(mOwningStorage).back());
        node = std::move(newNode);
    }
}
template <typename TemplateNodeType>
void Graph::initVisitsOf(std::unordered_map<size_t, inner::NodeState>& visits) const {
    auto& nodes = std::get<std::vector<TemplateNodeType>>(mOwningStorage);
    for (auto& node : nodes) {
        visits[node.getNodeIndex()] = inner::NodeState{0};
    }
}
void Graph::fullClear() {
    clear();
    mGraphIndex = inner::kKUndefinedIndex;
}
const SyncStorage& Graph::getSyncStorage() const {
    return mSyncStorage;
}
const OwningStorage& Graph::getOwningStorage() const {
    return mOwningStorage;
}
const Topology& Graph::getTopology() const {
    return mTopology;
}
void Graph::addNode(AbstractNode &node) {
    if (Graph* graphPointer = inner::getGraphTable()[node.getGraphIndex()]; graphPointer) {
        FatalError(1, "addNode() in Graph : ", this, ". GraphIndex : ", mGraphIndex, ". Adding Node to the graph to which it does not belong");
    }
    mSyncStorage.insert(node.getNodeIndex());
    inner::setGraphIndex(node, mGraphIndex);
}
void Graph::saveNode(AbstractNode &node, bool isRepairedNode, bool isErase) {
    if (node.getGraphIndex() != mGraphIndex) {
        FatalError(1, "saveNode() in Graph : ", this, ". GraphIndex : ", mGraphIndex, ". Saving Node in the graph to which it does not belong");
    }
    if (node.getType() == NodeType::DEFAULT) {
        saveRealNode(static_cast<Node&>(node), isRepairedNode, isErase);
    } else if (node.getType() == NodeType::INPUT) {
        saveRealNode(static_cast<InputNode&>(node), isRepairedNode, isErase);
    } else {
        FatalError(1, "saveNode() in Graph : ", this, ". GraphIndex : ", mGraphIndex, ". Undefined node type");
    }
}
void Graph::saveNode(AbstractNode &node, bool isRepairedNode) {
    saveNode(node, isRepairedNode, true);
}
void Graph::saveAllSyncNodes(bool isRepairedNode) {
    for (auto syncNode : mSyncStorage) {
        saveNode(*inner::getNodeTable()[syncNode], isRepairedNode, false);
    }
    mSyncStorage.clear();
}
void Graph::removeNode(AbstractNode &node) {
    mSyncStorage.erase(node.getNodeIndex());
    size_t nodeIndex = node.getNodeIndex();
    auto removePredicate = [nodeIndex](const inner::Edge& edge) -> bool {
        return nodeIndex == edge.startNodeIndex || nodeIndex == edge.endNodeIndex;
    };
    mTopology.erase(std::remove_if(mTopology.begin(), mTopology.end(), removePredicate), mTopology.end());
}
void Graph::link(const AbstractNode &startNode, const AbstractNode &endNode, EdgeMark mark) {
    if (endNode.getType() == NodeType::INPUT) {
        FatalError(1, "link() in Graph : ", this, ". GraphIndex : ", mGraphIndex, ". End node of edge can not be InputNode");
    }
    if (startNode.getGraphIndex() == endNode.getGraphIndex() && startNode.getGraphIndex() == mGraphIndex) {
        mTopology.emplace_back(startNode.getNodeIndex(), endNode.getNodeIndex(), mark);
        inner::incrementInputCount(*inner::getNodeTable()[endNode.getNodeIndex()]);
    } else {
        FatalError(1, "link() in Graph : ", this, ". GraphIndex : ", mGraphIndex, ". Nodes belong to different graphs");
    }
}
void Graph::clear() {
    mSyncStorage.clear();
    mTopology.clear();
    OwningStorage emptyStorage;
    mOwningStorage.swap(emptyStorage);
}
size_t Graph::countOwningNodes() const {
    return std::get<std::vector<Node>>(mOwningStorage).size()
        + std::get<std::vector<InputNode>>(mOwningStorage).size();
}
size_t Graph::countSyncNodes() const {
    return mSyncStorage.size();
}
size_t Graph::getGraphIndex() const {
    return mGraphIndex;
}
Traversal Graph::traverse(bool isRepairedNodes) {
    Traversal traversal;
    saveAllSyncNodes(isRepairedNodes);
    std::sort(mTopology.begin(), mTopology.end());
    std::queue<size_t> currentQueue, newQueue;
    std::unordered_map<size_t, inner::NodeState> visits;
    initVisitsOf<InputNode>(visits);    // TODO Is this need ?
    initVisitsOf<Node>(visits);
    for (auto& inputNode : std::get<std::vector<InputNode>>(mOwningStorage)) {
        currentQueue.push(inputNode.getNodeIndex());
    }
    while (true) {
        inner::Cluster cluster{0};
        while (!currentQueue.empty()) {
            size_t nodeIndex = currentQueue.front();
            currentQueue.pop();
            inner::Edge target(nodeIndex, 0, 0);
            auto edgeIterator = std::lower_bound(mTopology.begin(), mTopology.end(), target);
            while (edgeIterator != mTopology.end() && edgeIterator->startNodeIndex == nodeIndex) {
                auto& inputCount = visits[edgeIterator->endNodeIndex].inputCount;
                ++inputCount;
                auto targetInputCount = inner::getNodeTable()[edgeIterator->endNodeIndex]->getInputsCount();
                if (inputCount == targetInputCount) {
                    newQueue.push(edgeIterator->endNodeIndex);
                } else if (inputCount > targetInputCount) {
                    FatalError(1, "traverse() in Graph: ", mGraphIndex, ". Graph is have an cycle(s)");
                }
                visits[edgeIterator->endNodeIndex].input.emplace_back(nodeIndex, edgeIterator->mark);
                visits[nodeIndex].output.emplace_back(edgeIterator->endNodeIndex, edgeIterator->mark);
                ++edgeIterator;
            }
            AbstractNode *node = inner::getNodeTable()[nodeIndex];
            inner::setGraphIndex(*node, inner::kKUndefinedIndex);
            if (node->getType() == NodeType::INPUT) {
                cluster.get<InputNode>().emplace_back(
                    std::move(*static_cast<InputNode*>(node)),
                    std::move(visits[nodeIndex].input), std::move(visits[nodeIndex].output));
            } else if (node->getType() == NodeType::DEFAULT) {
                cluster.get<Node>().emplace_back(
                    std::move(*static_cast<Node*>(node)),
                    std::move(visits[nodeIndex].input), std::move(visits[nodeIndex].output));
            }
            ++cluster.nodeCount;
        }
        if (cluster.nodeCount > 0) {
            traversal.clusters.emplace_back(std::move(cluster));
        }
        std::swap(currentQueue, newQueue);
        if (currentQueue.empty()) {
            break;
        }
    }
    clear();
    return traversal;
}
}
