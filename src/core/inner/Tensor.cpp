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

#include <athena/core/inner/Tensor.h>
#include <athena/core/FatalError.h>
#include <athena/core/inner/GlobalTables.h>

#include <string>

namespace athena::core::inner {
Tensor::Tensor(const Tensor& rhs) : mShapeOffset(0), mAddressOffset(0), mShapePartialProduct(1) {
    mRecordIndex = inner::getAllocationTable().registerRecord(rhs.getDataType(),
        rhs.getShapeView().toShape());
}
Tensor::Tensor(DataType dataType, TensorShape shape)
    : mRecordIndex(inner::getAllocationTable().registerRecord(dataType, std::move(shape))),
      mShapeOffset(0), mAddressOffset(0), mShapePartialProduct(1) {
}
Tensor::Tensor(size_t recordIndex, size_t shapeOffset, size_t addressOffset, size_t shapePartialProduct)
    : mRecordIndex(recordIndex), mShapeOffset(shapeOffset), mAddressOffset(addressOffset),
      mShapePartialProduct(shapePartialProduct) {
}
Tensor &Tensor::operator=(const Tensor& rhs) {
    mShapeOffset = 0;
    mAddressOffset = 0;
    mRecordIndex = inner::getAllocationTable().registerRecord(rhs.getDataType(),
        rhs.getShapeView().toShape());
    mShapePartialProduct = 1;
    return *this;
}
Tensor Tensor::operator[](size_t index) const {
    auto shapeView = getAllocationTable()[mRecordIndex].getShapeView();
    size_t subShapePartialProduct = shapeView.dim(mShapeOffset) * mShapePartialProduct;
    return Tensor(mRecordIndex, mShapeOffset + 1, mAddressOffset + shapeView.getTotalSize()
        / subShapePartialProduct * index, subShapePartialProduct);
}
DataType Tensor::getDataType() const {
    return getAllocationTable()[mRecordIndex].getDataType();
}
ShapeView Tensor::getShapeView() const {
    return getAllocationTable()[mRecordIndex].getShapeView(mShapeOffset);
}
ShapeView Tensor::getSubShapeView(size_t offset) const {
    return getAllocationTable()[mRecordIndex].getShapeView(mShapeOffset + offset);
}
size_t Tensor::getAddress() const {
    return mAddressOffset;
}
void Tensor::clear() {
    mRecordIndex = inner::kKUndefinedIndex;
}
}  // namespace athena::core
