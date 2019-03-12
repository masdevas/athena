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

#include <string>
#include <stack>
#include <utility>
#include "AbstractGenerator.h"
#include "Tensor.h"

namespace athena::core {
    class Operation {
     protected:
        std::string mName;
     public:
        explicit Operation(std::string&& name) : mName(std::move(name)) {};
        virtual Tensor* getResultSize(std::deque<Tensor*> args) const = 0;
        virtual void gen(AbstractGenerator &g, std::stack<Tensor*> &operationArguments) const = 0;
        std::string getName();
    };
}

#endif //ATHENA_OPERATION_H
