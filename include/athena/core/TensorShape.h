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

#ifndef ATHENA_TENSORSHAPE_H
#define ATHENA_TENSORSHAPE_H

#include <utility>
#include <vector>

namespace athena::core {
class TensorShape {
 private:
    std::vector<size_t> mShape;
 public:
    explicit TensorShape(std::vector<size_t> shape) : mShape(std::move(shape)) {};
    TensorShape(const TensorShape &rhs);
    TensorShape(const TensorShape &&rhs) noexcept;

    /**
     * Returns shape as std::vector
     * @return vector containing sizes for every dimension
     */
    const std::vector<size_t> &getShape() const;

    /**
     * Returns number of elements in Tensor
     * @return number of elements in Tensor
     */
    size_t getTotalSize() const;

    /**
     * Returns size for certain dimension
     * @param index Dimension index ( 0 <= index < dimensions )
     * @return Size for dimension
     */
    size_t dim(size_t index) const;

    /**
     * Returns number of dimensions in the shape
     * @return Number of dimensions
     */
    size_t dimensions() const;

    /**
     * Returns clone of mShape without first element
     * @return Clone of mShape without first element
     */
    TensorShape subshape() const;
};
}
#endif //ATHENA_TENSORSHAPE_H
