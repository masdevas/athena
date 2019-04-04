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

#ifndef ATHENA_INPUTNODE_H
#define ATHENA_INPUTNODE_H

#include <athena/core/AbstractNode.h>
#include <athena/core/AbstractLoader.h>

namespace athena::core {
class InputNode : public AbstractNode {
 private:
    const AbstractLoader *mLoader;
 public:
    InputNode() = delete;
    InputNode(const InputNode& rhs) = default;
    InputNode(InputNode&& rhs) noexcept = default;
    InputNode(TensorShape shape, DataType dataType,
              const AbstractLoader& loader, std::string name = "");
    ~InputNode() override;

    InputNode &operator=(const InputNode& rhs) = default;
    InputNode &operator=(InputNode&& rhs) noexcept = default;

    NodeType getType() const override;
    const AbstractLoader& getLoader() const;
    void clear() override;
};
}

#endif //ATHENA_INPUTNODE_H
