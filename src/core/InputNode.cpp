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

#include <athena/core/InputNode.h>

namespace athena::core {
InputNode::InputNode(TensorShape shape, DataType dataType,
                     const AbstractLoader& loader, std::string name)
    : AbstractNode(std::move(shape), dataType, std::move(name)), mLoader(&loader)  {
}
InputNode::~InputNode() {
    saveInGraph(false);
}
NodeType InputNode::getType() const {
    return NodeType::INPUT;
}
const AbstractLoader& InputNode::getLoader() const {
    return *mLoader;
}
void InputNode::clear() {
    AbstractNode::clear();
    mLoader = nullptr;
}
}