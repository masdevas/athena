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

void AddOperation::gen(
    core::AbstractGenerator &g,
    std::vector<core::inner::Tensor *> &operationArguments) const {
    core::inner::Tensor *c = operationArguments[2];
    core::inner::Tensor *b = operationArguments[1];
    core::inner::Tensor *a = operationArguments[0];

    g.generate("add", *a, *b, *c);
}
core::inner::Tensor *AddOperation::getResultTensor(
    std::vector<core::inner::Tensor *> args) const {
    core::ShapeView shapeView(args[0]->getShapeView());
    core::inner::Tensor *res =
        new core::inner::Tensor(args[0]->getDataType(), shapeView.toShape());
    return res;
}

core::inner::Tensor *AddOperation::getDerivativeTensor(
    std::vector<core::inner::Tensor *> args, int argNo) const {
#ifdef DEBUG
    assert(argNo < 2 && "AddOperation takes 2 arguments!");
#endif
    core::ShapeView shapeView(args[argNo]->getShapeView());
    core::inner::Tensor *res =
        new core::inner::Tensor(args[argNo]->getDataType(), shapeView.toShape());
    return res;
}
void AddOperation::genDerivative(
    core::AbstractGenerator &g,
    std::vector<core::inner::Tensor *> &operationArguments,
    int argNo) const {
    // todo implementation
}

}  // namespace athena::ops
