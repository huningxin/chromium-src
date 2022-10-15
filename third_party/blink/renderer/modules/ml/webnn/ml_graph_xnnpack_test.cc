// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder_test.h"

#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/blink/renderer/bindings/core/v8/native_value_traits_impl.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_tester.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_binding_for_testing.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_dom_exception.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_xnnpack.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"

namespace blink {

class MLGraphXnnpackTest : public testing::Test {
 public:
  MLGraphXnnpackTest() = default;
  ~MLGraphXnnpackTest() override = default;
};

MLGraphXnnpack* ToMLGraphXnnpack(V8TestingScope* scope, ScriptValue value) {
  return NativeValueTraits<MLGraphXnnpack>::NativeValue(
      scope->GetIsolate(), value.V8Value(), scope->GetExceptionState());
}

TEST_F(MLGraphXnnpackTest, SharedXnnpackContextTest) {
  V8TestingScope scope;
  auto* builder = CreateMLGraphBuilder(scope);
  auto* script_state = scope.GetScriptState();
  {
    // Test building MLGraphXnnpack and creating pthreadpool successfully.
    auto* input = BuildInput(scope, builder, "input", {3, 4, 5},
                             V8MLOperandType::Enum::kFloat32);
    auto* output = builder->relu(input, scope.GetExceptionState());
    EXPECT_NE(output, nullptr);

    ScriptPromiseTester tester(
        script_state, builder->buildAsync(script_state, {{"output", output}},
                                          scope.GetExceptionState()));
    tester.WaitUntilSettled();
    EXPECT_TRUE(tester.IsFulfilled());
    auto* xnnpack_graph = ToMLGraphXnnpack(&scope, tester.Value());
    EXPECT_NE(xnnpack_graph, nullptr);
    const auto& inputs = xnnpack_graph->GetInputResourcesInfo();
    EXPECT_EQ(inputs.size(), static_cast<uint32_t>(1));
    EXPECT_EQ(inputs.at("input").type, input->Type());
    EXPECT_EQ(inputs.at("input").byte_length, input->ByteLength());
    const auto& outputs = xnnpack_graph->GetOutputResourcesInfo();
    EXPECT_EQ(outputs.size(), static_cast<uint32_t>(1));
    EXPECT_EQ(outputs.at("output").type, output->Type());
    EXPECT_EQ(outputs.at("output").byte_length, output->ByteLength());
    EXPECT_NE(xnnpack_graph->GetPthreadpoolForTesting(), nullptr);
  }
  {
    // Test building two MLGraphXnnpack instances that share the same
    // pthreadpool.
    auto* input = BuildInput(scope, builder, "input", {3, 4, 5},
                             V8MLOperandType::Enum::kFloat32);
    auto* output1 = builder->relu(input, scope.GetExceptionState());
    EXPECT_NE(output1, nullptr);
    ScriptPromiseTester tester1(
        script_state, builder->buildAsync(script_state, {{"output", output1}},
                                          scope.GetExceptionState()));
    tester1.WaitUntilSettled();
    EXPECT_TRUE(tester1.IsFulfilled());
    auto* xnnpack_graph1 = ToMLGraphXnnpack(&scope, tester1.Value());
    EXPECT_NE(xnnpack_graph1, nullptr);

    auto* output2 = builder->hardSwish(input, scope.GetExceptionState());
    EXPECT_NE(output2, nullptr);
    ScriptPromiseTester tester2(
        script_state, builder->buildAsync(script_state, {{"output", output2}},
                                          scope.GetExceptionState()));
    tester2.WaitUntilSettled();
    EXPECT_TRUE(tester2.IsFulfilled());
    auto* xnnpack_graph2 = ToMLGraphXnnpack(&scope, tester2.Value());
    EXPECT_NE(xnnpack_graph2, nullptr);

    EXPECT_EQ(xnnpack_graph1->GetPthreadpoolForTesting(),
              xnnpack_graph2->GetPthreadpoolForTesting());
  }
  {
    // Test building MLGraphXnnpack and initializing XNNPACK library
    // successfully.
    auto* input = BuildInput(scope, builder, "input", {3, 4, 5},
                             V8MLOperandType::Enum::kFloat32);
    auto* output = builder->relu(input, scope.GetExceptionState());
    EXPECT_NE(output, nullptr);

    ScriptPromiseTester tester(
        script_state, builder->buildAsync(script_state, {{"output", output}},
                                          scope.GetExceptionState()));
    tester.WaitUntilSettled();
    EXPECT_TRUE(tester.IsFulfilled());
    auto* xnnpack_graph = ToMLGraphXnnpack(&scope, tester.Value());
    EXPECT_NE(xnnpack_graph, nullptr);

    // Test the XNNPACK initialization and memory allocator by creating a clamp
    // op.
    xnn_operator_t xnn_clamp_op = nullptr;
    xnn_status status = xnn_create_clamp_nc_f32(
        3, 3, 3, 0.0f, std::numeric_limits<float>::infinity(), 0,
        &xnn_clamp_op);
    EXPECT_EQ(status, xnn_status_success);
    EXPECT_NE(xnn_clamp_op, nullptr);
    status = xnn_delete_operator(xnn_clamp_op);
    EXPECT_EQ(status, xnn_status_success);
  }
}

}  // namespace blink
