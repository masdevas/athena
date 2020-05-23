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

#ifndef ATHENA_TUPLECONTAINERS_H
#define ATHENA_TUPLECONTAINERS_H

#include <athena/core/core_export.h>
#include <athena/utils/Index.h>

#include <tuple>

namespace athena::utils::internal {
//template <template <typename, typename...> typename Container,
//          template <typename> typename Wrapper, typename... Args>
//struct ATH_CORE_EXPORT TupleContainersWithWrappers {
//  using Holder = std::tuple<Container<Wrapper<Args>>...>;
//};
template <template <typename, typename...> typename Map,
    template <typename> typename Value, typename... Args>
struct ATH_CORE_EXPORT TupleMaps {
  using Holder = std::tuple<Map<Index, Value<Args>>...>;
};

template <template <typename, typename...> typename Container, typename... Args>
struct ATH_CORE_EXPORT TupleContainers {
  using Holder = std::tuple<Container<Args>...>;
};
} // namespace athena::core::internal

#endif // ATHENA_TUPLECONTAINERS_H
