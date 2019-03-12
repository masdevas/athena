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
#ifndef ATHENA_FATALERROR_H
#define ATHENA_FATALERROR_H
#include "Logger.h"

#include <athena/core/Error.h>
#include <athena/core/log.h>

#include <iostream>
#include <string_view>

namespace athena::core {
class FatalError : public Error {
    public:
    explicit FatalError(std::string_view error) : Error(error) {
        err() << mErrorMessage;
        exit(1);
    }
};
}  // namespace athena::core

#endif  // ATHENA_FATALERROR_H
