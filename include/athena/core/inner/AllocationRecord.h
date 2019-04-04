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

#ifndef ATHENA_ALLOCATIONTABLERECORD_H
#define ATHENA_ALLOCATIONTABLERECORD_H

#include <athena/core/DataType.h>
#include <athena/core/TensorShape.h>

namespace athena::core::inner {
struct AllocationRecord {
 private:
    DataType mDataType;
    TensorShape mShape;

 public:
    AllocationRecord(DataType dataType, TensorShape shape);
    DataType getDataType() const;
    ShapeView getShapeView(size_t offset = 0) const;
};
}

#endif //ATHENA_ALLOCATIONTABLERECORD_H
