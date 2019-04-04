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
Node::Node(TensorShape shape, DataType dataType, const Operation &operation, std::string name)
    : AbstractNode(std::move(shape), dataType, std::move(name)), mOperation(&operation) {
}
Node::~Node() {
    saveInGraph(false);
}
NodeType Node::getType() const {
    return NodeType::DEFAULT;
}
const Operation& Node::getOperation() const {
    return *mOperation;
}
void Node::clear() {
    AbstractNode::clear();
    mOperation = nullptr;
}
}