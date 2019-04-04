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

#ifndef ATHENA_ABSTRACTNODE_H
#define ATHENA_ABSTRACTNODE_H

#include <athena/core/DataType.h>
#include <athena/core/NodeType.h>
#include <athena/core/Clear.h>
#include <athena/core/inner/Tensor.h>
#include <athena/core/inner/IndexFunctions.h>

#include <string>
#include <string_view>

namespace athena::core {
using EdgeMark = size_t;

class AbstractNode {
 protected:
    inner::Tensor mTensor;
    std::string mName;
    size_t mGraphIndex;
    size_t mNodeIndex;
    size_t mInputsCount;
    void fullClear();
 public:
    AbstractNode() = delete;
    AbstractNode(const AbstractNode& rhs);
    AbstractNode(AbstractNode&& rhs) noexcept;
    AbstractNode(TensorShape shape, DataType dataType, std::string name);
    virtual ~AbstractNode();

    AbstractNode &operator=(const AbstractNode& rhs);
    AbstractNode &operator=(AbstractNode&& rhs) noexcept;

    void after(const AbstractNode& node, EdgeMark mark) const;
    void before(const AbstractNode& node, EdgeMark mark) const;
    ShapeView getShapeView() const;
    ShapeView getSubShapeView(size_t offset = 1) const;
    DataType getDataType() const;
    virtual NodeType getType() const = 0;
    size_t getNodeIndex() const;
    size_t getGraphIndex() const;
    size_t getInputsCount() const;
    std::string_view getName() const;
    std::string& name();
    void removeFromGraph();
    void saveInGraph(bool isRepairedNode = true);
    virtual void clear();
    friend void inner::setGraphIndex(AbstractNode &node, size_t graphIndex);
    friend void inner::incrementInputCount(athena::core::AbstractNode& node);
};
}

#endif //ATHENA_ABSTRACTNODE_H
