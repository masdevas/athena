#include <athena/core/node/internal/AbstractNodeInternal.h>
#include <athena/core/node/internal/NodeInternal.h>
#include <athena/operation/internal/FillOperationInternal.h>

namespace athena::operation::internal {
FillOperationInternal::FillOperationInternal(
    utils::SharedPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, core::DataType type, core::TensorShape shape,
    float pattern, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)),
      mDataType(type), mShape(shape), mFloatPattern(pattern){};
FillOperationInternal::FillOperationInternal(
    utils::SharedPtr<core::internal::ContextInternal> context,
    utils::Index publicNodeIndex, core::DataType type, core::TensorShape shape,
    double pattern, utils::String name)
    : core::internal::OperationInternal(std::move(context), publicNodeIndex,
                                        std::move(name)),
      mDataType(type), mShape(shape), mDoublePattern(pattern){};
utils::Index FillOperationInternal::createResultTensor(
    utils::SharedPtr<core::internal::ContextInternal> context,
    std::vector<core::internal::TensorInternal*> tensorIndexes) const {
  return context->create<core::internal::TensorInternal>(
      context, context->getNextPublicIndex(), mDataType, mShape);
}

core::internal::GenValue FillOperationInternal::gen(
    utils::SharedPtr<core::internal::ContextInternal> context,
    core::internal::Generator& generator,
    std::vector<utils::Index>& operationArguments,
    core::internal::GenNode parent) const {
  // TODO be batch-aware.
  generator.callBuiltin<core::internal::builtin::Lock>(
      parent.getResult(), core::internal::LockType::READ_WRITE);

  // TODO other data types
  core::internal::GenValue val;
  if (mDataType == core::DataType::FLOAT) {
    val = generator.createConstant(mFloatPattern);
  }
  auto result = generator.callBuiltin<core::internal::builtin::Fill>(
      val, parent.getResult());

  generator.callBuiltin<core::internal::builtin::Release>(parent.getResult());

  return result;
}

std::tuple<utils::Index, std::vector<core::internal::Edge>,
           std::vector<utils::Index>>
FillOperationInternal::genDerivative(
    const core::NodeState* currentNodeState, size_t indexOfOutputDependence,
    utils::Index gradientGraphFinalNodeIndex) const {
  new utils::FatalError(utils::ATH_FATAL_OTHER, "Not implemented");
  return std::make_tuple<utils::Index, std::vector<core::internal::Edge>,
                         std::vector<utils::Index>>(0, {}, {});
}
size_t FillOperationInternal::getOperandsCount() const { return 0; }
} // namespace athena::operation::internal
