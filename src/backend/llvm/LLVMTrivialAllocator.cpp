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

#include <athena/backend/llvm/LLVMTrivialAllocator.h>
#include <athena/core/DataType.h>

#include <memory>

namespace athena::backend::llvm {
void LLVMTrivialAllocator::allocate(const athena::core::Tensor& tensor) {
    auto mem = reinterpret_cast<void*>(mAllocator.allocate(
        tensor.getShape().getTotalSize() *
        athena::core::sizeOfDataType(tensor.getDataType())));
    mAllocatedMap[tensor.getVirtualAddress()] = mem;
}
size_t LLVMTrivialAllocator::getRAMPointer() {
    return 0;
}
size_t LLVMTrivialAllocator::getFastPointer(const athena::core::Tensor& t) {
    return (size_t)mAllocatedMap[t.getVirtualAddress()];
}
}  // namespace athena::backend::llvm
