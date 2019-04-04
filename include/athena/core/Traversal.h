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

#ifndef ATHENA_TRAVERSAL_H
#define ATHENA_TRAVERSAL_H

#include <athena/core/inner/TupleContainers.h>
#include <athena/core/Node.h>
#include <athena/core/InputNode.h>

namespace athena::core {
namespace inner {
struct Dependence {
    size_t nodeIndex;
    size_t mark;
    Dependence(size_t nodeIndex, size_t mark)
        : nodeIndex(nodeIndex), mark(mark) {
    }
};
template <typename TemplateNodeType>
struct NodeDependencies {
    TemplateNodeType node;
    std::vector<Dependence> input;
    std::vector<Dependence> output;
    NodeDependencies(TemplateNodeType node, std::vector<Dependence> input, std::vector<Dependence> output)
        : node(std::move(node)), input(std::move(input)), output(std::move(output)) {
    }
};

struct NodeState {
    size_t inputCount;
    std::vector<Dependence> input;
    std::vector<Dependence> output;
};

struct Cluster {
    using Content = inner::TupleContainersWithWrappers<std::vector, inner::NodeDependencies, Node, InputNode>::Holder;
    size_t nodeCount;
    Content content;
    template <typename TemplateNodeType>
    std::vector<inner::NodeDependencies<TemplateNodeType>>& get() {
        return std::get<std::vector<inner::NodeDependencies<TemplateNodeType>>>(content);
    }
};
}

struct Traversal {
    std::vector<inner::Cluster> clusters;
};
}

#endif  // ATHENA_TRAVERSAL_H
