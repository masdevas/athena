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

#include <athena/core/Node.h>

size_t athena::core::Node::mNodeCounter = 0;

athena::core::Node::Node(Node&& src) noexcept : mIncomingNodes(std::move(src.mIncomingNodes)),
                                                mOutgoingNodes(std::move(src.mOutgoingNodes)),
                                                mOperation(std::move(src.mOperation)),
                                                mName(std::move(src.mName)) {
}

athena::core::Node::Node(athena::core::Operation&& op) : mOperation(std::move(op)),
                                                         mName(mOperation.getName() +
                                                               std::to_string(++mNodeCounter)) {
}

athena::core::Node& athena::core::Node::operator=(Node&& src) noexcept {
    mIncomingNodes = std::move(src.mIncomingNodes);
    mOutgoingNodes = std::move(src.mOutgoingNodes);
    mOperation = std::move(src.mOperation);
    mName = std::move(src.mName);
    return *this;
}

void athena::core::Node::after(athena::core::Node* node) {
    node->mOutgoingNodes.push_back(this);
    mIncomingNodes.push_back(node);
}