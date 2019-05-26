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

#include <athena/core/FatalError.h>

#include <gtest/gtest.h>

using namespace athena::core;

TEST(FatalError, FatalErrorAbortsExecution) {
#ifdef DEBUG
    ASSERT_DEBUG_DEATH(new FatalError(6, "FatalError test"), "FatalError test");
#else
    EXPECT_EXIT(new FatalError(3, "FatalError test"),
                ::testing::ExitedWithCode(3), "FatalError test");
#endif
}