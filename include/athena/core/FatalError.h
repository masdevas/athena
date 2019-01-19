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
#include <string_view>
#include <athena/core/Error.h>
#include <athena/core/log.h>
#include "Logger.h"
#include <iostream>


namespace athena::core {
class FatalError : public Error {
 public:
    explicit FatalError(std::string_view error) : Error(error) {
        err() << mErrorMessage;
        exit(1);
    }
};
}

#endif //ATHENA_FATALERROR_H
