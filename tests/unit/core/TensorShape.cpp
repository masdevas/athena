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

#include <gtest/gtest.h>
#include <athena/core/TensorShape.h>

namespace athena::core {
TEST(TensorShapeTest, Creation) {
    TensorShape({1, 2, 3});
}

TEST(TensorShapeTest, Dimension) {
    TensorShape shape({1, 2, 3, 4});
    ASSERT_EQ(shape.dimensions(), 4);
}

TEST(TensorShapeTest, TotalSize) {
    TensorShape shape({1, 2, 3, 4});
    ASSERT_EQ(shape.getTotalSize(), 1 * 2 * 3 * 4);
}
}