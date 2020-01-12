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

#ifndef ATHENA_TENSOR_ALLOCATOR_H
#define ATHENA_TENSOR_ALLOCATOR_H

#include <athena/core/core_export.h>

#include <cstddef>

namespace athena::core::internal {
/**
 * Interface used by backend to manage memory
 */
class ATH_CORE_EXPORT TensorAllocator {
public:
  virtual ~TensorAllocator() = default;
  virtual void allocate(const TensorInternal&) = 0;
  virtual size_t getRAMPointer(const TensorInternal&) = 0;
  virtual size_t getFastPointer(const TensorInternal&) = 0;
};

} // namespace athena::core::internal

#endif // ATHENA_TENSOR_ALLOCATOR_H
