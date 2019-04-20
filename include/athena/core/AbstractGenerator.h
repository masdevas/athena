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
#ifndef ATHENA_ABSTRACTGENERATOR_H
#define ATHENA_ABSTRACTGENERATOR_H

#include <athena/core/inner/Tensor.h>

namespace athena::core {

class AbstractGenerator {
    protected:
    virtual void generateImpl(std::string &, inner::Tensor &a) = 0;
    virtual void generateImpl(std::string &,
                              inner::Tensor &a,
                              inner::Tensor &b) = 0;
    virtual void generateImpl(std::string &,
                              inner::Tensor &a,
                              inner::Tensor &b,
                              inner::Tensor &c) = 0;

    public:
    virtual void openNode(std::string_view name) = 0;
    virtual void closeNode() = 0;
    template <typename... Args>
    void generate(std::string name, Args &&... a) {
        generateImpl(name, std::forward<Args>(a)...);
    };
};

}  // namespace athena::core

#endif  // ATHENA_ABSTRACTGENERATOR_H
