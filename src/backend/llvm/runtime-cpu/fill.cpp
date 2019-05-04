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

#include <athena/backend/llvm/runtime/fill.h>
#include <athena/core/Allocator.h>
#include <athena/core/inner/Tensor.h>

extern "C" {
    void ffill(void *allocator, void *tensor, float f) {
        auto *pAllocator =
            reinterpret_cast<athena::core::Allocator *>(allocator);
        auto *pTensor = reinterpret_cast<athena::core::inner::Tensor *>(tensor);

        auto *mem = reinterpret_cast<float *>(pAllocator->getRAMPointer(*pTensor));

        for (size_t i = 0; i < pTensor->getShapeView().getTotalSize(); i++) {
            mem[i] = f;
        }
    }
}
