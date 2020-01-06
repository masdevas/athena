---
id: core_optimizer
title: Core: Optimizer
---

# Athena Graph Optimizer design
Abstract base class `Optimizer` provides this list of methods for 
overriding by inheritors:
- `genForward`
- `genBackward`

## Forward pass 
Call of this function create function in IR that is computing computational 
graph value. That is not a pure virtual function, so function might not be overridden
in inheritors.

### Operands:

1. `generator`: generator for IR.
2. `graph`: target computational graph.

TBD

### Results:

None

## Backward pass
Call of this function create function in IR that is changing data in 
non-frozen input nodes of graph accordance to optimizing algorithm,
that is represented by the optimizer.  

### Operands:

1. `generator`: generator for IR.
2. `graph`: target computational graph.

TBD

### Results:

None
