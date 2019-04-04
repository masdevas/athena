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

#include <athena/core/inner/IndexFunctions.h>
#include <athena/core/AbstractNode.h>

namespace athena::core::inner {
void setGraphIndex(AbstractNode &node, size_t graphIndex) {
    node.mGraphIndex = graphIndex;
}
void incrementInputCount(AbstractNode &node) {
    ++node.mInputsCount;
}
}