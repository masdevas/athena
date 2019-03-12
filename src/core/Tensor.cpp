/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#include <athena/core/Tensor.h>

namespace athena::core {
Tensor& Tensor::operator[](size_t index) {
    if (index > mShape.dimensions()) {
        FatalError("Out of range index in Tensor: " + std::to_string(reinterpret_cast<size_t>(this)));
    }
    size_t newVirtualAddress = mVirtualAddress +
        mShape.getTotalSize() / mShape.dim(0) * index;
    return *(new Tensor(mDataType, mShape.subshape(), newVirtualAddress));
}

DataType Tensor::getDataType() const {
    return mDataType;
}

size_t Tensor::getVirtualAddress() const {
    return mVirtualAddress;
}

const TensorShape& Tensor::getShape() const {
    return mShape;
}
void Tensor::setVirtualAddress(size_t address) {
    if (mVirtualAddress == 0) // tensors can not be relocated
        mVirtualAddress = address;
}
}