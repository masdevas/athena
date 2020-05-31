#ifndef ATHENA_FILLOPERATIONINTERNAL_H
#define ATHENA_FILLOPERATIONINTERNAL_H

#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/operation/internal/OperationInternal.h>
#include <athena/operation/operation_export.h>
#include <athena/utils/allocator/Allocator.h>

namespace athena::operation::internal {
class ATH_OPERATION_EXPORT FillOperationInternal
    : public core::internal::OperationInternal {
public:
  FillOperationInternal(
      utils::SharedPtr<core::internal::ContextInternal> context,
      utils::Index publicNodeIndex, core::DataType, core::TensorShape shape,
      float pattern, utils::String name = utils::String(""));
  FillOperationInternal(
      utils::SharedPtr<core::internal::ContextInternal> context,
      utils::Index publicNodeIndex, core::DataType, core::TensorShape shape,
      double pattern, utils::String name = utils::String(""));

  ~FillOperationInternal() = default;

  [[nodiscard]] utils::Index
  createResultTensor(utils::SharedPtr<core::internal::ContextInternal> context,
                     std::vector<core::internal::TensorInternal*> tensorIndexes)
      const override;

  core::internal::GenValue
  gen(utils::SharedPtr<core::internal::ContextInternal> context,
      core::internal::Generator& generator,
      std::vector<utils::Index>& operationArguments,
      core::internal::GenNode parentNode) const override;

  // output node and edges of generated graph
  std::tuple<utils::Index, std::vector<core::internal::Edge>,
             std::vector<utils::Index>>
  genDerivative(const core::NodeState* currentNodeState,
                size_t indexOfOutputDependence,
                utils::Index gradientGraphFinalNodeIndex) const override;

  [[nodiscard]] virtual size_t getOperandsCount() const override;

private:
  core::DataType mDataType;
  core::TensorShape mShape;
  float mFloatPattern;
  double mDoublePattern;
};
} // namespace athena::operation::internal

#endif // ATHENA_FILLOPERATIONINTERNAL_H
