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

#ifndef ATHENA_CONTEXT_H
#define ATHENA_CONTEXT_H

#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/core_export.h>
#include <athena/utils/Index.h>
#include <athena/utils/allocator/Allocator.h>
#include <athena/core/ForwardDeclarations.h>
#include <athena/core/Wrapper.h>

namespace athena::core {
class ATH_CORE_EXPORT Context {
public:
  explicit Context(utils::Allocator allocator = utils::Allocator(), size_t defaultCapacity = 1, size_t elementAverageSize = 1);

  explicit Context(utils::SharedPtr<internal::ContextInternal> ptr);

  ~Context();

  template <typename Type, typename... Args>
  typename Wrapper<Type>::PublicType create(Args&&... args);

  utils::Allocator& getAllocator();

  internal::ContextInternal* internal();

  const internal::ContextInternal* internal() const;

public:
  utils::SharedPtr<internal::ContextInternal> mInternal;
};

template <typename Type, typename... Args>
typename Wrapper<Type>::PublicType Context::create(Args&&... args) {
  auto index = mInternal->create<typename Type::InternalType>(mInternal, mInternal->getNextPublicIndex(),
      std::forward<Args>(args)...);
  return Returner<Type>::returner(mInternal, index);
}

//class ATH_CORE_EXPORT Context {
//public:
//  friend inner::Table<AbstractNode*>&
//  inner::getNodeTable(athena::core::Context& context);
//  friend inner::Table<inner::AllocationRecord>&
//  inner::getAllocationTable(athena::core::Context& context);
//  friend inner::Table<Graph*>&
//  inner::getGraphTable(athena::core::Context& context);
//
//private:
//  inner::Table<inner::AllocationRecord> allocationTable;
//  inner::Table<AbstractNode*> nodeTable;
//  inner::Table<Graph*> graphTable;
//};
} // namespace athena::core

#endif // ATHENA_CONTEXT_H
