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

#include <athena/core/AbstractNode.h>

namespace athena::core {

size_t AbstractNode::mNodeCounter = 0;

AbstractNode::AbstractNode(AbstractNode &&node) noexcept
    : mOutgoingNodes(std::move(node.mOutgoingNodes)), mName(std::move(node.mName)) {
}

AbstractNode::AbstractNode(std::string&& name)
    : mName(std::move(name)) {
}

AbstractNode& AbstractNode::operator=(AbstractNode &&src) noexcept {
    mOutgoingNodes = std::move(src.mOutgoingNodes);
    mName = std::move(src.mName);
    return *this;
}

void AbstractNode::addOutgoingNode(AbstractNode* node) {
    mOutgoingNodes.emplace_back(node);
}

}