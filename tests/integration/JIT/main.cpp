#include <athena/backend/llvm/LLVMExecutor.h>
#include <athena/core/context/Context.h>
#include <athena/core/graph/Graph.h>
#include <athena/core/node/InputNode.h>
#include <athena/core/node/Node.h>
#include <athena/core/node/OutputNode.h>
#include <athena/operation/AddOperation.h>
#include <athena/operation/FillOperation.h>

#include <gtest/gtest.h>

using namespace athena;
using namespace athena::core;
using namespace athena::operation;
using namespace athena::backend::llvm;

TEST(JITIntegration, FillOperationSample) {
  Context context;
  auto graph = context.create<Graph>("testGraph");

  auto inp1 =
      graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false, 0, "inp1");
  auto inp2 =
      graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false, 0, "inp2");
  auto operationId = context.create<AddOperation>();
  auto node1 = graph.create<Node>(operationId, "node1");
  graph.connect(inp1, node1, AddOperation::LEFT);
  graph.connect(inp2, node1, AddOperation::RIGHT);

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  SUCCEED();
}
