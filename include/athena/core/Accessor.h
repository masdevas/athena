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

#ifndef ATHENA_ACCESSOR_H
#define ATHENA_ACCESSOR_H

#include <athena/core/Allocator.h>
#include <athena/core/inner/Tensor.h>

#include <utility>
namespace athena::core {

template <typename T>
class Accessor {
    private:
    Allocator &mAllocator;
    inner::Tensor &mAccessTensor;
    std::vector<size_t> mAccessIdx;

    public:
    Accessor(Allocator &allocator,
             inner::Tensor &tensor,
             std::vector<size_t> idx)
        : mAllocator(allocator),
          mAccessTensor(tensor),
          mAccessIdx(std::move(idx)) {}

    Accessor<T> operator[](size_t idx) {
        if (mAccessIdx.size() == mAccessTensor.getShapeView().dimensions() /*||
            idx >= mAccessTensor.getShapeView().dim(mAccessIdx.size())*/) {
            new FatalError(1, "Index is out of range");
        }

        std::vector<size_t> newIdx(mAccessIdx);
        newIdx.push_back(idx);

        return Accessor<T>(mAllocator, mAccessTensor, newIdx);
    }

    T operator*() {
        if (mAccessIdx.size() != mAccessTensor.getShapeView().dimensions()) {
            new FatalError(2, "Accessor does not point to an element");
        }
        size_t offset = 0;
        for (int d = 0; d < mAccessIdx.size() - 1; d++) {
            offset += mAccessIdx[d] * mAccessTensor.getShapeView().dim(d);
        }
        offset += mAccessIdx.back();

        auto elPtr =
            reinterpret_cast<T *>(mAllocator.getRAMPointer(mAccessTensor));

        return *(elPtr + offset);
    }
};
}  // namespace athena::core

#endif  // ATHENA_ACCESSOR_H
