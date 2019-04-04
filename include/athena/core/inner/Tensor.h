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

#ifndef ATHENA_TENSOR_H
#define ATHENA_TENSOR_H

#include <athena/core/inner/AllocationRecord.h>

namespace athena::core::inner {
class Tensor {
    private:
    using RegisteredTensors = std::vector<inner::AllocationRecord>;
    size_t mRecordIndex;
    size_t mShapeOffset;
    size_t mAddressOffset;
    size_t mShapePartialProduct;
    Tensor(size_t recordIndex, size_t shapeOffset, size_t addressOffset,
        size_t shapePartialProduct);

    public:
    Tensor() = delete;
    Tensor(const Tensor& rhs);
    Tensor(Tensor&& rhs) noexcept = default;
    Tensor(DataType dataType, TensorShape shape);
    ~Tensor() = default;

    Tensor& operator=(const Tensor& rhs);
    Tensor& operator=(Tensor&& rhs) noexcept = default;
    /**
     * Returns subtensor like a new object
     * @return Reference to new Tensor on the same memory
     */
    Tensor operator[](size_t index) const;

    DataType getDataType() const;
    ShapeView getShapeView() const;
    ShapeView getSubShapeView(size_t offset = 1) const;
    size_t getAddress() const;
    void clear();
};
}  // namespace athena::core

#endif  // ATHENA_TENSOR_H
