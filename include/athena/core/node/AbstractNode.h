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

#ifndef ATHENA_ABSTRACTNODE_H
#define ATHENA_ABSTRACTNODE_H

#include <athena/core/core_export.h>
#include <athena/core/PublicEntity.h>

namespace athena::core {
namespace internal {
class AbstractNodeInternal;
}

/**
 * A Node represents a piece of computation in Graph
 */
class ATH_CORE_EXPORT AbstractNode : public PublicEntity {
public:
  using InternalType = internal::AbstractNodeInternal;
};

} // namespace athena::core

#endif // ATHENA_ABSTRACTNODE_H