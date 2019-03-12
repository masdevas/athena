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
#ifndef ATHENA_ERROR_H
#define ATHENA_ERROR_H

#include <ostream>
#include <string_view>

namespace athena::core {

class Error {
    protected:
    std::string_view mErrorMessage;
    bool mHasError;

    public:
    Error() : mErrorMessage(), mHasError(false){};
    explicit Error(std::string_view message)
        : mErrorMessage(message), mHasError(true){};
    Error(const Error &error)     = default;
    Error(Error &&error) noexcept = default;
    Error &operator=(const Error &error) = default;
    Error &operator=(Error &&error) noexcept = default;
    ~Error()                                 = default;

    operator bool() const;
    friend std::ostream &operator<<(std::ostream &stream, const Error &err);
};
}  // namespace athena::core

#endif  // ATHENA_ERROR_H
