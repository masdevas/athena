/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#include <athena/core/Error.h>

namespace athena::core {
Error::Error() : mErrorCode(0) {}
Error::operator bool() const {
    return mErrorCode != 0;
}
std::ostream &operator<<(std::ostream &stream, const Error &err) {
    stream << err.mErrorMessage << "\n";
    return stream;
}
}  // namespace athena::core