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

#include <cstddef>
#include <memory>
#include "DataType.h"
#include "TensorShape.h"

namespace athena::core {
class Tensor {
 private:
    DataType mDataType;
    size_t mVirtualAddress;
    TensorShape mShape;
 public:
    Tensor(DataType dataType, size_t virtualAddress, TensorShape shape)
        : mDataType(dataType),
          mVirtualAddress(virtualAddress),
          mShape(std::move(shape)) {}
    Tensor(const Tensor& rhs) = default;
    Tensor(Tensor&& rhs) noexcept = default;
    ~Tensor() = default;

    Tensor& operator=(const Tensor& rhs) = default;
    Tensor& operator=(Tensor&& rhs) noexcept = default;
    /**
     * Returns subtensor like a new object
     * @return Reference to new Tensor on the same memory
     */
    Tensor& operator[](size_t index);

    DataType getDataType() const;
    size_t getVirtualAddress() const;
    const TensorShape& getShape() const;
};
}

#endif //ATHENA_TENSOR_H
