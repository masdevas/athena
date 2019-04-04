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

#ifndef ATHENA_OPERATION_H
#define ATHENA_OPERATION_H

#include <athena/core/AbstractGenerator.h>
#include <athena/core/inner/Tensor.h>
#include <athena/core/FatalError.h>

#include <stack>
#include <string>
#include <utility>

namespace athena::core {
class Operation {
    protected:
    std::string mName;

    public:
    explicit Operation(std::string&& name) : mName(std::move(name)){};
    virtual inner::Tensor* getResultTensor(
        std::deque<inner::Tensor*> args) const   = 0;
    virtual void gen(AbstractGenerator& g,
                     std::stack<inner::Tensor*>& operationArguments) const = 0;
    std::string getName() const;
};

class OperationDummy : public Operation {
 public:
    explicit OperationDummy(std::string name) : Operation(std::move(name)){};

    [[noreturn]] inner::Tensor* getResultTensor(
        std::deque<inner::Tensor*> args) const override {
        FatalError(1, "NOT IMPL");
    }

    void gen(AbstractGenerator& g,
             std::stack<inner::Tensor*>& operationArguments) const override {
        FatalError(1, "NOT IMPL");
    }
};
}  // namespace athena::core

#endif  // ATHENA_OPERATION_H
