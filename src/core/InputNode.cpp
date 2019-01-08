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

#include "athena/core/InputNode.h"

namespace athena::core {

InputNode::InputNode(InputNode &&node) noexcept
    : AbstractNode(std::move(node)), tensor(node.tensor) {
    node.tensor = nullptr;
}

InputNode& InputNode::operator=(InputNode&& src) noexcept {
    mOutgoingNodes = std::move(src.mOutgoingNodes);
    tensor = src.tensor;
    mName = std::move(src.mName);
    src.tensor = nullptr;
    return *this;
}

void InputNode::after(AbstractNode* node) {
    //throw "Error. Input node can not be after something!";
}

}