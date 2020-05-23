/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_DATATYPE_H
#define ATHENA_DATATYPE_H

#include <athena/core/core_export.h>

#include <cstddef>

namespace athena::core {
enum class DataType : int { UNDEFINED = 0, DOUBLE = 1, FLOAT = 2, HALF = 3 };

ATH_CORE_EXPORT size_t sizeOfDataType(const DataType& dataType);
} // namespace athena::core

#endif // ATHENA_DATATYPE_H
