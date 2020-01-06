---
id: core_operation
title: Core: Operation
---

# Athena Operations
Abstract base class `Operation` provides this list of methods for 
overriding by inheritors:
- `genCall`
- `genDerivative`

### Notes
Let's call node which contains considered operation's object like a "current node".

## genCall 
Call of this function will add lines to generating IR that are a sequence of
actions for calculating considered operation value by arguments.

### Operands:

1. `generator`: generator for IR.
2. `operationArguments`: tensors containing arguments for current node.

### Results:

None

## genDerivative 
Call of this function will create computational graph for derivative
of operation in current node by one of inputs.

### Operands:

1. `operationArguments`: tensors containing arguments for operation.
2. `derivativeOfIncomingNode`: tensor of incoming node for derivative of this node by incoming node.
3. `ownDerivative`: tensor for derivative of outgoing nodes by this node.
3. `argumentMark`: mark of argument for derivative calculation.

### Results:

Graph that is representation derivative of operation by target input.
