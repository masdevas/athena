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

#include <athena/core/inner/Tensor.h>

#include <numeric>
#include <gtest/gtest.h>

//TODO Test copy semantic

namespace athena::core::inner {

void testTensorCreation(const std::vector<size_t> &data, DataType dataType) {
    auto shape = TensorShape(data);
    auto tensor = Tensor(dataType, shape);
    ASSERT_EQ(dataType, tensor.getDataType());
    ASSERT_EQ(shape, tensor.getShapeView());
    std::vector resultShapeViewData(shape.getShapeView().begin(), shape.getShapeView().end());
    ASSERT_EQ(resultShapeViewData, data);
    if (data.size() > 0) {
        ASSERT_EQ(TensorShape(data.begin() + 1, data.end()), tensor.getSubShapeView());
        std::vector resultSubShapeViewData(shape.getSubShapeView().begin(),
                                           shape.getSubShapeView().end());
        std::vector expectedSubShapeViewData(data.begin() + 1, data.end());
        ASSERT_EQ(expectedSubShapeViewData, expectedSubShapeViewData);
    }
    ASSERT_EQ(0, tensor.getAddress());
}

TEST(TensorTest, Creation) {
    {
        auto dataType = DataType::DOUBLE;
        std::vector<size_t> data{2, 5, 8, 12};
        testTensorCreation(data, dataType);
    }
    {
        auto dataType = DataType::DOUBLE;
        std::vector<size_t> data{2, 5, 21, 8, 12};
        testTensorCreation(data, dataType);
    }
    {
        auto dataType = DataType::DOUBLE;
        std::vector<size_t> data{8, 12};
        testTensorCreation(data, dataType);
    }
    {
        auto dataType = DataType::DOUBLE;
        std::vector<size_t> data{8};
        testTensorCreation(data, dataType);
    }
}

TEST(TensorTest, CreationSeveralTensors) {
    auto dataType = DataType::DOUBLE;
    std::vector<size_t> data{2, 5, 8, 12};
    testTensorCreation(data, dataType);
    std::vector<size_t> data_second{2, 5, 21, 8, 12};
    testTensorCreation(data_second, dataType);
    std::vector<size_t> data_third{2, 5, 21, 8, 12};
    testTensorCreation(data_third, dataType);
    std::vector<size_t> data_fourth{};
    testTensorCreation(data_fourth, dataType);
    std::vector<size_t> data_fifth{2, 5, 21, 8, 12};
    testTensorCreation(data_fifth, dataType);
}

TEST(TensorTest, GetSubTensor) {
    auto dataType = DataType::DOUBLE;
    std::vector<size_t> data{50, 65, 8, 12};
    auto shape = TensorShape(data);
    auto tensor = Tensor(dataType, shape);
    for (size_t indexSubTensor = 0; indexSubTensor < data[0]; ++indexSubTensor) {
        auto subTensor = tensor[indexSubTensor];
        ASSERT_EQ(subTensor.getDataType(), dataType);
        ASSERT_TRUE(subTensor.getShapeView() == TensorShape(data.begin() + 1, data.end()));
        size_t subSize = std::accumulate<std::vector<size_t>::const_iterator, size_t>(data.begin() + 1,
                                                                                      data.end(),
                                                                                      1,
                                                                                      std::multiplies<size_t>());
        ASSERT_EQ(subTensor.getAddress(), indexSubTensor * subSize);
    }
}

TEST(TensorTest, GetSubSubTensor) {
    auto dataType = DataType::DOUBLE;
    std::vector<size_t> data{10, 10, 10};
    auto shape = TensorShape(data);
    auto tensor = Tensor(dataType, shape);
    size_t subSize = std::accumulate<std::vector<size_t>::const_iterator, size_t>(data.begin() + 1,
                                                                                  data.end(),
                                                                                  1,
                                                                                  std::multiplies<size_t>());
    size_t subSubSize = std::accumulate<std::vector<size_t>::const_iterator, size_t>(data.begin() + 2,
                                                                                  data.end(),
                                                                                  1,
                                                                                  std::multiplies<size_t>());
    for (size_t indexSubTensor = 0; indexSubTensor < data[0]; ++indexSubTensor) {
        auto subTensor = tensor[indexSubTensor];
        for (size_t deepIndex = 0; deepIndex < data[1]; ++deepIndex) {
            auto subSubTensor = subTensor[deepIndex];
            ASSERT_EQ(subSubTensor.getDataType(), dataType);
            ASSERT_EQ(subSubTensor.getShapeView(), TensorShape(data.begin() + 2, data.end()));
            ASSERT_EQ(subSubTensor.getAddress(), subSize * indexSubTensor + subSubSize * deepIndex);
        }
    }
}

TEST(TensorTest, CopyConstructor) {
    auto dataType = DataType::HALF;
    std::vector<size_t> data{10, 12, 10};
    auto shape = TensorShape(data);
    auto tensor = Tensor(dataType, shape);
    size_t subSize = std::accumulate<std::vector<size_t>::const_iterator, size_t>(data.begin() + 1,
                                                                                  data.end(),
                                                                                  1,
                                                                                  std::multiplies<size_t>());
    {
        auto tensorCopy = tensor;
        ASSERT_EQ(tensorCopy.getDataType(), dataType);
        ASSERT_EQ(tensorCopy.getShapeView(), shape);
        ASSERT_EQ(tensorCopy.getAddress(), 0);
    }
    {
        for (size_t index = 0; index < shape[0]; ++index) {
            auto subTensor = tensor[index];
            ASSERT_EQ(subTensor.getDataType(), dataType);
            ASSERT_EQ(subTensor.getShapeView(), shape.getSubShapeView());
            ASSERT_EQ(subTensor.getAddress(), subSize * index);
        }
    }
}

TEST(TensorTest, CopyOperator) {
    auto dataType = DataType::HALF;
    std::vector<size_t> data{10, 12, 10};
    auto shape = TensorShape(data);
    auto tensor = Tensor(dataType, shape);
    {
        auto dataTypeSecond = DataType::HALF;
        std::vector<size_t> dataSecond{10, 12, 10};
        auto shapeSecond = TensorShape(data);
        auto tensorSecond = Tensor(dataTypeSecond, shapeSecond);
        tensor = tensorSecond;
        ASSERT_EQ(tensor.getDataType(), dataTypeSecond);
        ASSERT_EQ(tensor.getShapeView(), shapeSecond);
        ASSERT_EQ(tensor.getAddress(), 0);
    }
    {
        for (size_t index = 0; index < shape[0]; ++index) {
            auto subTensor = tensor[index];
            auto tensorSecond = subTensor;
            Tensor tensorThird(dataType, shape);
            tensorThird = subTensor;
            ASSERT_EQ(tensorSecond.getDataType(), dataType);
            ASSERT_EQ(tensorSecond.getShapeView(), subTensor.getShapeView());
            ASSERT_EQ(tensorSecond.getAddress(), 0);
            ASSERT_EQ(tensorThird.getDataType(), dataType);
            ASSERT_EQ(tensorThird.getShapeView(), subTensor.getShapeView());
            ASSERT_EQ(tensorThird.getAddress(), 0);
        }
    }
}
}  // namespace athena::core
