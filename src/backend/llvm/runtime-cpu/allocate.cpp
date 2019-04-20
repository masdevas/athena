/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#include <athena/backend/llvm/runtime/allocate.h>
#include <athena/core/Allocator.h>
#include <athena/core/inner/Tensor.h>

extern "C" {
void allocate(void *allocator, void *tensor) {
    auto pAllocator = reinterpret_cast<athena::core::Allocator *>(allocator);
    auto pTensor = reinterpret_cast<athena::core::inner::Tensor *>(tensor);

    pAllocator->allocate(*pTensor);
}

size_t get_fast_pointer(void *allocator, void *tensor) {
    auto pAllocator = reinterpret_cast<athena::core::Allocator *>(allocator);
    auto pTensor = reinterpret_cast<athena::core::inner::Tensor *>(tensor);

    return pAllocator->getFastPointer(*pTensor);
}
}