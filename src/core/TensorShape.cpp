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

#include <athena/core/TensorShape.h>
#include <string>
#include <stdexcept>
#include <athena/core/FatalError.h>

namespace athena::core {
TensorShape::TensorShape(const TensorShape &rhs) {
    mShape = rhs.mShape;
}
TensorShape::TensorShape(const TensorShape &&rhs) noexcept {
    mShape = rhs.mShape;
}
const std::vector<size_t> &TensorShape::getShape() const {
    return mShape;
}
size_t TensorShape::getTotalSize() const {
    size_t totalSize = 1;

    for (auto size : mShape) {
        totalSize *= size;
    }

    return mShape.empty() ? 0 : totalSize;
}
size_t TensorShape::dim(size_t index) const {
    if (index >= mShape.size()) {
        FatalError("TensorShape only has " + std::to_string(mShape.size()) + " dimensions. "
            + std::to_string(index) + " requested.");
    }
    return mShape[index];
}
size_t TensorShape::dimensions() const {
    return mShape.size();
}
TensorShape TensorShape::subshape() const {
    return TensorShape(std::vector<size_t>(mShape.begin() + 1, mShape.end()));
}
}