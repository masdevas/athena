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

#ifndef ATHENA_NODETYPE_H
#define ATHENA_NODETYPE_H

namespace athena::core {
enum class NodeType { UNDEFINED, INPUT, DEFAULT };

template <typename TemplateNodeType>
NodeType getNodeType();
}  // namespace athena::core

#endif  // ATHENA_NODETYPE_H
