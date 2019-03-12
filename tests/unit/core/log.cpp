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

#include "athena/core/log.h"

#include <athena/core/TensorShape.h>

#include <gtest/gtest.h>
#include <string>

namespace athena::core {
TEST(TensorLog, SetCout) { setLogStream<Logger>(std::cout); }
TEST(TensorLog, SetStringstream) {
    std::stringstream ss;
    setLogStream<Logger>(ss);
    std::string firstPart = "Hello", secondPart = "Log";
    std::string fullString = firstPart + secondPart;
    log() << firstPart << secondPart;
    ASSERT_EQ(ss.str(), fullString);
}
}  // namespace athena::core