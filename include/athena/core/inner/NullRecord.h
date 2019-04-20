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

#ifndef ATHENA_NULLRECORD_H
#define ATHENA_NULLRECORD_H

#include <athena/core/FatalError.h>
#include <athena/core/inner/AllocationRecord.h>
#include <athena/core/inner/ForwardDeclarations.h>

namespace athena::core::inner {
template <typename Content>
struct NullRecord;
template <>
struct NullRecord<AllocationRecord> {
    static inline const AllocationRecord value =
        AllocationRecord(DataType::UNDEFINED, TensorShape{});
};
template <>
struct NullRecord<Graph*> {
    static inline Graph* const value = nullptr;
};
template <>
struct NullRecord<AbstractNode*> {
    static inline AbstractNode* const value = nullptr;
};
}  // namespace athena::core::inner

#endif  // ATHENA_NULLRECORD_H
