/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#include <athena/backend/llvm/runtime/Device.h>
#include <athena/backend/llvm/runtime/add.h>
#include <athena/backend/llvm/runtime/fill.h>
#include <athena/core/loader/internal/TensorAllocator.h>
#include <athena/core/tensor/impl/TensorImpl.h>

using namespace athena::backend::llvm;
using namespace athena::core::inner;
using namespace athena::core;

template <typename T>
void fill(Device*, Allocator* allocator, Tensor* a, T value) {
  auto* memory = reinterpret_cast<T*>(allocator->getRAMPointer(*a));

  for (size_t i = 0; i < a->getShape().getTotalSize(); i++) {
    memory[i] = value;
  }
}

template void fill<float>(athena::backend::llvm::Device*,
                          athena::core::Allocator*,
                          athena::core::inner::Tensor* a, float value);

template void fill<double>(athena::backend::llvm::Device*,
                           athena::core::Allocator*,
                           athena::core::inner::Tensor* a, double value);
