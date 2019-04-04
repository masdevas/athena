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

#include <athena/ops/AddOperation.h>

namespace athena::ops {

void AddOperation::gen(core::AbstractGenerator &g,
                       std::stack<core::inner::Tensor *> &operationArguments) const {
    core::inner::Tensor *c = operationArguments.top();
    operationArguments.pop();
    core::inner::Tensor *b = operationArguments.top();
    operationArguments.pop();
    core::inner::Tensor *a = operationArguments.top();
    operationArguments.pop();

    g.generateAdd(*a, *b, *c);
}
core::inner::Tensor *AddOperation::getResultTensor(
    std::deque<core::inner::Tensor *> args) const {
    core::ShapeView shapeView(args[0]->getShapeView());
    core::inner::Tensor *res = new core::inner::Tensor(args[0]->getDataType(), shapeView.toShape());
    return res;
}

}  // namespace athena::ops
