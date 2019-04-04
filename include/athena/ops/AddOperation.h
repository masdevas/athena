/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#ifndef ATHENA_ADDOPERATION_H
#define ATHENA_ADDOPERATION_H

#include <athena/core/AbstractGenerator.h>
#include <athena/core/Operation.h>
#include <athena/core/inner/Tensor.h>

namespace athena::ops {
class AddOperation : public core::Operation {
    public:
    AddOperation() : Operation("add") {}

    core::inner::Tensor *getResultTensor(
        std::deque<core::inner::Tensor *> args) const override;
    void gen(core::AbstractGenerator &g,
             std::stack<core::inner::Tensor *> &operationArguments) const override;
};
}  // namespace athena::ops
#endif  // ATHENA_ADDOPERATION_H
