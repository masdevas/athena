#ifndef ATHENA_FILLOPERATION_H
#define ATHENA_FILLOPERATION_H

#include <athena/core/operation/Operation.h>
#include <athena/operation/internal/FillOperationInternal.h>
#include <athena/operation/operation_export.h>

namespace athena::operation {
class ATH_OPERATION_EXPORT FillOperation : public core::Operation {
public:
  using InternalType = internal::FillOperationInternal;
};
}

#endif // ATHENA_FILLOPERATION_H
