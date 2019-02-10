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
#include <athena/core/Tensor.h>

namespace athena::core {
TEST(TensorTest, Creation) {
    auto dataType = DataType::DOUBLE;
    size_t virtualAddress = 12;
    auto shape = TensorShape({2, 5, 8, 12});
    auto tensor = Tensor({dataType, virtualAddress, shape});
    ASSERT_EQ(dataType, tensor.getDataType());
    ASSERT_EQ(virtualAddress, tensor.getVirtualAddress());
    ASSERT_EQ(shape.getShape(), tensor.getShape().getShape());
}

TEST(TensorTest, OperatorIndexing) {
    auto dataType = DataType::DOUBLE;
    size_t virtualAddress = 12;
    auto shape = TensorShape({2, 5, 8, 12});
    auto tensor = Tensor({dataType, virtualAddress, shape});
    auto subtensor = tensor[50];
}
}