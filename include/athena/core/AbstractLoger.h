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
#ifndef ATHENA_ABSTRACTLOGER_H
#define ATHENA_ABSTRACTLOGER_H

#include <string_view>
#include <string>
#include <athena/core/Error.h>

namespace athena::core {

class Error;

class AbstractLogger {
 public:
    AbstractLogger() = default;
    AbstractLogger(const AbstractLogger &) = default;
    AbstractLogger(AbstractLogger &&) noexcept = default;
    AbstractLogger &operator=(const AbstractLogger &) = default;
    AbstractLogger &operator=(AbstractLogger &&) noexcept = default;
    virtual ~AbstractLogger() = default;

    template<typename T>
    AbstractLogger &operator<<(const T &data) {
        return streamImpl(data);
    }

 protected:
    virtual AbstractLogger &streamImpl(const std::string &data) = 0;
    virtual AbstractLogger &streamImpl(const std::string_view &data) = 0;
    virtual AbstractLogger &streamImpl(const Error &data) = 0;
    virtual AbstractLogger &streamImpl(const char *data) = 0;
};

}
#endif //ATHENA_ABSTRACTLOGER_H
