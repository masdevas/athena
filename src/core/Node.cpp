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

namespace athena::core {
Node::Node(Node&& rhs) noexcept
    : AbstractNode(std::move(rhs)), mOperation(rhs.mOperation) {
    rhs.fullClear();
}
Node::Node(TensorShape shape,
           DataType dataType,
           Operation& operation,
           std::string name)
    : AbstractNode(std::move(shape), dataType, std::move(name)),
      mOperation(&operation) {}
Node::~Node() {
    saveInGraph(false);
}
Node& Node::operator=(Node&& rhs) noexcept {
    AbstractNode::operator=(std::move(rhs));
    mOperation = std::move(rhs.mOperation);
    rhs.fullClear();
    return *this;
}
void Node::fullClear() {
    Node::clear();
}
NodeType Node::getType() const {
    return NodeType::DEFAULT;
}
const Operation& Node::getOperation() const {
    return *mOperation;
}
const Operation* Node::getOperationPtr() const {
    return mOperation;
}
Operation& Node::operation() {
    return *mOperation;
}
void Node::setOperation(Operation& operation) {
    mOperation = &operation;
}
void Node::clear() {
    mOperation = nullptr;
    AbstractNode::clear();
}
}  // namespace athena::core