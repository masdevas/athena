/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_NODETYPE_H
#define ATHENA_NODETYPE_H

#include <athena/core/FatalError.h>
#include <athena/core/inner/ForwardDeclarations.h>

namespace athena::core {
enum class NodeType {
    UNDEFINED = 0,
    INPUT = 1,
    DEFAULT = 2,
    OUTPUT = 3,
    LOSS = 4
};

template <typename TemplateNodeType>
NodeType getNodeType();

template <NodeType Type>
struct NodeTypeId {};

template <>
struct NodeTypeId<NodeType::DEFAULT> : std::decay<Node> {};

template <>
struct NodeTypeId<NodeType::INPUT> : std::decay<InputNode> {};

template <>
struct NodeTypeId<NodeType::LOSS> : std::decay<LossNode> {};

template <>
struct NodeTypeId<NodeType::OUTPUT> : std::decay<OutputNode> {};
}  // namespace athena::core

namespace athena {
template <typename NodeTypeName, typename ParentNodeType>
typename std::enable_if<std::is_reference<NodeTypeName>::value,
                        NodeTypeName>::type
node_cast(ParentNodeType &node) {
    using PureType = typename std::remove_reference<NodeTypeName>::type;
#ifdef DEBUG
    if (node.getType() != core::getNodeType<PureType>()) {
        new core::FatalError(127, "Attempt to cast incompatible types");
    }
#endif

    return *reinterpret_cast<PureType *>(&node);
}

template <typename NodeTypeName, typename ParentNodeType>
typename std::enable_if<std::is_pointer<NodeTypeName>::value,
                        NodeTypeName>::type
node_dyncast(ParentNodeType node) {
    using PureType = typename std::remove_pointer<NodeTypeName>::type;
#ifdef DEBUG
    if (node->getType() != core::getNodeType<PureType>()) {
        new core::FatalError(127, "Attempt to cast incompatible types");
    }
#endif

    return reinterpret_cast<PureType *>(node);
}
}  // namespace athena

#endif  // ATHENA_NODETYPE_H
