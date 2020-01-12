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

#ifndef ATHENA_DECREASEOPERATION_H
#define ATHENA_DECREASEOPERATION_H

#include <athena/operation/internal/DecreaseOperationInternal.h>
#include <athena/core/operation/Operation.h>

namespace athena::operation {
class ATH_OPERATION_EXPORT DecreaseOperation : public core::Operation {
public:
using InternalType = internal::DecreaseOperationInternal;
enum Arguments {
  Unmarked
};
};
}

#endif // ATHENA_DECREASEOPERATION_H
