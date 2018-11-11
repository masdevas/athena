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
athena::core::Node::Node(athena::core::Operation &&op) : mOperation(op) {
    mName = mOperation.getName() + std::to_string(++mNodeCounter);
}

void athena::core::Node::after(athena::core::Node &node) {
    node.mOutgoingNodes.push_back(this);
    mIncomingNodes.push_back(&node);
}
athena::core::Node::Node(const athena::core::Node &&src) noexcept : mOperation(src.mOperation) {
    mIncomingNodes = src.mIncomingNodes;
    mOutgoingNodes = src.mOutgoingNodes;
    mName = src.mName;
}