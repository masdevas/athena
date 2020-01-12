/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/core/node/internal/AbstractNodeInternal.h>
#include <athena/core/node/internal/NodeInternal.h>
#include <athena/loaders/internal/ConstantLoaderInternal.h>
#include <athena/operation/MulOperation.h>
#include <athena/operation/internal/DecreaseOperationInternal.h>

namespace athena::operation::internal {
DecreaseOperationInternal::DecreaseOperationInternal(
    utils::SharedPtr<core::internal::ContextInternal> context, utils::Index publicNodeIndex, double multiplier, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex, std::move(name)), mMultiplier(multiplier) {}

utils::Index DecreaseOperationInternal::createResultTensor(utils::SharedPtr<core::internal::ContextInternal> context,
                                                      std::vector<core::internal::TensorInternal*> tensorIndexes) const {
  auto dataType = tensorIndexes[0]->getDataType();
  auto tensorShape = tensorIndexes[0]->getShape();
  //TODO check preconditions
  return context->create<core::internal::TensorInternal>(context, context->getNextPublicIndex(), dataType, std::move(tensorShape));
}

void DecreaseOperationInternal::gen(utils::SharedPtr<core::internal::ContextInternal> context, core::Generator& generator,
                 std::vector<utils::Index>& operationArguments) const {
  //TODO call generator
}

std::tuple<utils::Index, std::vector<core::internal::Edge>, std::vector<utils::Index>> DecreaseOperationInternal::genDerivative(const core::NodeState* currentNodeState,
                                                                                                size_t indexOfOutputDependence, utils::Index gradientGraphFinalNodeIndex) const {
  // TODO
  return {};
}

size_t DecreaseOperationInternal::getOperandsCount() const {
  return 1;
}
}
