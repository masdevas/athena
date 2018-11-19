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
athena::core::InputNode::InputNode(athena::core::InputNode &&src) noexcept : mTensor(src.mTensor), Node(std::move(src)) {}
athena::core::InputNode &athena::core::InputNode::operator=(athena::core::InputNode &&src) noexcept {
    mIncomingNodes = std::move(src.mIncomingNodes);
    mOutgoingNodes = std::move(src.mOutgoingNodes);
    mOperation = std::move(src.mOperation);
    mName = std::move(src.mName);
    mTensor = src.mTensor;
    return *this;
}
