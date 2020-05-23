/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_MSE_H
#define ATHENA_MSE_H

#include <athena/backend/llvm/runtime/Device.h>
#include <athena/core/loader/internal/TensorAllocator.h>
#include <athena/core/tensor/impl/TensorImpl.h>

template <typename T>
extern void mse(athena::backend::llvm::Device*, athena::core::Allocator*,
                athena::core::inner::Tensor* a, athena::core::inner::Tensor* b,
                athena::core::inner::Tensor* c);

#endif // ATHENA_MSE_H
