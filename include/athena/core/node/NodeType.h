/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://getathena.ml
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

#include <athena/core/core_export.h>
#include <athena/core/ForwardDeclarations.h>
#include <athena/utils/error/FatalError.h>
#include <type_traits>

namespace athena::core {
enum class ATH_CORE_EXPORT NodeType {
  UNDEFINED = 0,
  INPUT = 1,
  DEFAULT = 2,
  OUTPUT = 3
};

//template <typename TemplateNodeType> ATH_CORE_EXPORT NodeType getNodeType();

} // namespace athena::core

//namespace athena {
//template <typename NodeTypeName, typename ParentNodeType>
//ATH_CORE_EXPORT NodeTypeName node_static_cast(ParentNodeType& node) {
//  using PureType = typename std::decay<NodeTypeName>::type;
//#ifdef DEBUG
//  if (node.getType() != core::getNodeType<PureType>()) {
//    new utils::FatalError(utils::ATH_BAD_CAST,
//                         "Attempt to cast incompatible types");
//  }
//#endif
//  return *reinterpret_cast<PureType*>(&node);
//}
//} // namespace athena

#endif // ATHENA_NODETYPE_H
