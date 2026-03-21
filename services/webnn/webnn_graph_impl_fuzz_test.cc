// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <limits>
#include <string>
#include <vector>

#include "base/at_exit.h"
#include "base/base_switches.h"
#include "base/command_line.h"
#include "base/compiler_specific.h"
#include "base/containers/flat_map.h"
#include "base/test/bind.h"
#include "base/test/run_until.h"
#include "base/test/scoped_feature_list.h"
#include "base/test/test_future.h"
#include "base/test/test_timeouts.h"
#include "base/types/expected_macros.h"
#include "build/build_config.h"
#include "base/test/allow_check_is_test_for_testing.h"
#include "mojo/core/embedder/embedder.h"
#include "mojo/public/cpp/bindings/associated_remote.h"
#include "mojo/public/cpp/bindings/remote.h"
#include "services/webnn/buildflags.h"
#include "services/webnn/public/cpp/context_properties.h"
#include "services/webnn/public/cpp/graph_validation_utils.h"
#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/public/cpp/webnn_types.h"
#include "services/webnn/public/mojom/features.mojom-features.h"
#include "services/webnn/public/mojom/webnn_context.mojom.h"
#include "services/webnn/public/mojom/webnn_context_provider.mojom.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"
#include "services/webnn/public/mojom/webnn_graph_builder.mojom.h"
#include "services/webnn/public/mojom/webnn_tensor.mojom.h"
#include "services/webnn/webnn_context_impl.h"
#include "services/webnn/webnn_context_provider_impl.h"
#include "services/webnn/webnn_test_environment.h"
#include "services/webnn/webnn_test_utils.h"
#include "services/webnn/webnn_utils.h"
#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/fuzztest/src/fuzztest/domain_core.h"
#include "third_party/fuzztest/src/fuzztest/fuzztest.h"
#include "third_party/fuzztest/src/fuzztest/googletest_fixture_adapter.h"

#if BUILDFLAG(IS_MAC)
#include "base/mac/mac_util.h"
#endif  // BUILDFLAG(IS_MAC)

namespace webnn::test {

namespace {

#define ASSIGN_OR_RETURN_VOID(lhs, rexpr) \
  ASSIGN_OR_RETURN(lhs, rexpr, [](std::string error) { return; });

struct TensorRemoteAndHandle {
  mojo::AssociatedRemote<mojom::WebNNTensor> remote;
  blink::WebNNTensorToken handle;
};

TensorRemoteAndHandle CreateTensor(
    mojo::Remote<mojom::WebNNContext>& context_remote,
    mojom::TensorInfoPtr tensor_info) {
  mojo::AssociatedRemote<mojom::WebNNTensor> webnn_tensor_remote;

  base::test::TestFuture<mojom::CreateTensorResultPtr> create_tensor_future;
  context_remote->CreateTensor(std::move(tensor_info), mojo_base::BigBuffer(0),
                               create_tensor_future.GetCallback());
  mojom::CreateTensorResultPtr create_tensor_result =
      create_tensor_future.Take();
  EXPECT_TRUE(create_tensor_result->is_success());
  webnn_tensor_remote.Bind(
      std::move(create_tensor_result->get_success()->tensor_remote));
  EXPECT_TRUE(webnn_tensor_remote.is_bound());

  return TensorRemoteAndHandle{
      .remote = std::move(webnn_tensor_remote),
      .handle = create_tensor_result->get_success()->tensor_handle};
}

TensorRemoteAndHandle CreateTensorWithValues(
    mojo::Remote<mojom::WebNNContext>& context_remote,
    mojom::TensorInfoPtr tensor_info,
    base::span<const uint8_t> data) {
  auto remote_and_handle = CreateTensor(context_remote, std::move(tensor_info));
  remote_and_handle.remote->WriteTensor(mojo_base::BigBuffer(data));
  return remote_and_handle;
}

void BuildAndCompute(
    mojo::Remote<mojom::WebNNContext>& context_remote,
    mojo::AssociatedRemote<mojom::WebNNGraphBuilder> graph_builder_remote,
    mojom::GraphInfoPtr graph_info,
    base::flat_map<std::string, base::span<const uint8_t>> named_inputs) {
  // Create input tensors.
  std::vector<std::pair<std::string, TensorRemoteAndHandle>>
      named_input_remotes_and_handles;
  named_input_remotes_and_handles.reserve(graph_info->input_operands.size());

  for (OperandId operand_id : graph_info->input_operands) {
    const mojom::Operand& operand =
        *graph_info->operands.at(operand_id.value());
    EXPECT_TRUE(operand.name.has_value());

    auto it = named_inputs.find(*operand.name);
    EXPECT_TRUE(it != named_inputs.end());

    auto tensor_info = mojom::TensorInfo::New(
        operand.descriptor, MLTensorUsage{MLTensorUsageFlags::kWrite});
    named_input_remotes_and_handles.emplace_back(
        *operand.name, CreateTensorWithValues(
                           context_remote, std::move(tensor_info), it->second));
  }

  // Create output tensors.
  std::vector<std::pair<std::string, TensorRemoteAndHandle>>
      named_output_remotes_and_handles;
  named_output_remotes_and_handles.reserve(graph_info->output_operands.size());

  for (OperandId operand_id : graph_info->output_operands) {
    const mojom::Operand& operand =
        *graph_info->operands.at(operand_id.value());
    EXPECT_TRUE(operand.name.has_value());

    auto tensor_info = mojom::TensorInfo::New(
        operand.descriptor, MLTensorUsage{MLTensorUsageFlags::kRead});
    named_output_remotes_and_handles.emplace_back(
        *operand.name, CreateTensor(context_remote, std::move(tensor_info)));
  }

  base::test::TestFuture<
      base::expected<mojom::CreateGraphSuccessPtr, mojom::ErrorPtr>>
      create_graph_future;

  graph_builder_remote->CreateGraph(std::move(graph_info),
                                    create_graph_future.GetCallback());
  auto create_graph_result = create_graph_future.Take();
  if (!create_graph_result.has_value()) {
    LOG(ERROR) << "Failed to create graph: "
               << create_graph_result.error()->message;
    return;
  }

  mojo::AssociatedRemote<mojom::WebNNGraph> graph_remote;
  graph_remote.Bind(std::move(create_graph_result.value()->graph_remote));

  std::vector<std::pair<std::string, blink::WebNNTensorToken>>
      named_input_handles;
  named_input_handles.reserve(named_input_remotes_and_handles.size());
  std::ranges::transform(
      named_input_remotes_and_handles, std::back_inserter(named_input_handles),
      [](const auto& input) {
        return std::make_pair(input.first, input.second.handle);
      });

  std::vector<std::pair<std::string, blink::WebNNTensorToken>>
      named_output_handles;
  named_output_handles.reserve(named_output_remotes_and_handles.size());
  std::ranges::transform(
      named_output_remotes_and_handles,
      std::back_inserter(named_output_handles), [](const auto& output) {
        return std::make_pair(output.first, output.second.handle);
      });

  graph_remote->Dispatch(named_input_handles, named_output_handles);

  // Wait for the computation to complete.
  for (auto& output : named_output_remotes_and_handles) {
    base::test::TestFuture<mojom::ReadTensorResultPtr> read_tensor_future;
    output.second.remote->ReadTensor(read_tensor_future.GetCallback());
    std::ignore = read_tensor_future.Wait();
  }

  graph_remote.reset();
  graph_builder_remote.reset();
}

}  // namespace

class GlobalFuzzEnvironment {
 public:
  GlobalFuzzEnvironment() {
    base::test::AllowCheckIsTestForTesting();

    at_exit_manager_ = std::make_unique<base::AtExitManager>();

    TestTimeouts::Initialize();
    mojo::core::Init();

    webnn_test_environment_ = std::make_unique<WebNNTestEnvironment>();
  }

  WebNNTestEnvironment& GetWebNNTestEnvironment() {
    return *webnn_test_environment_;
  }

 private:
  std::unique_ptr<base::AtExitManager> at_exit_manager_;
  std::unique_ptr<WebNNTestEnvironment> webnn_test_environment_;
};

GlobalFuzzEnvironment& GetGlobalFuzzEnvironment() {
  static base::NoDestructor<GlobalFuzzEnvironment> instance;
  return *instance;
}

class WebNNGraphImplFuzzTestBase : public testing::Test {
 public:
  WebNNGraphImplFuzzTestBase()
      : context_properties_(GetContextPropertiesForTesting()) {}

  bool IsIncognito() const { return false; }
  void SetUp() override;
  void SetUpBase();
  void TearDown() override;

  mojo::AssociatedRemote<mojom::WebNNGraphBuilder> BindNewGraphBuilderRemote();

  mojo::Remote<mojom::WebNNContext>& context() { return webnn_context_; }

 protected:
  base::test::ScopedFeatureList scoped_feature_list_;

  ContextProperties context_properties_;

  mojo::Remote<mojom::WebNNContextProvider> provider_remote_;
  mojo::Remote<mojom::WebNNContext> webnn_context_;
};

void WebNNGraphImplFuzzTestBase::SetUp() {
#if BUILDFLAG(IS_MAC)
  if (base::mac::MacOSVersion() < 14'00'00) {
    GTEST_SKIP() << "Skipping test because WebNN is not supported on Mac OS "
                 << base::mac::MacOSVersion();
  }
#endif  // BUILDFLAG(IS_MAC)
  SetUpBase();
}

void WebNNGraphImplFuzzTestBase::SetUpBase() {
  GetGlobalFuzzEnvironment().GetWebNNTestEnvironment().BindWebNNContextProvider(
      provider_remote_.BindNewPipeAndPassReceiver(), IsIncognito());

  base::CommandLine* command_line = base::CommandLine::ForCurrentProcess();
  std::string enable_features =
      command_line->GetSwitchValueASCII(switches::kEnableFeatures);
  std::string disable_features =
      command_line->GetSwitchValueASCII(switches::kDisableFeatures);

  scoped_feature_list_.InitFromCommandLine(enable_features, disable_features);

  base::test::TestFuture<mojom::CreateContextResultPtr> create_context_future;
  provider_remote_->CreateWebNNContext(
      mojom::CreateContextOptions::New(
          mojom::Device::kCpu,
          mojom::CreateContextOptions::PowerPreference::kDefault),
      create_context_future.GetCallback());
  mojom::CreateContextResultPtr create_context_result =
      create_context_future.Take();
  if (create_context_result->is_success()) {
    webnn_context_.Bind(
        std::move(create_context_result->get_success()->context_remote));
    context_properties_ =
        create_context_result->get_success()->context_properties;
  } else {
    GTEST_SKIP() << "Failed to create WebNN context: "
                 << create_context_result->get_error()->message;
  }
}

void WebNNGraphImplFuzzTestBase::TearDown() {
  webnn_context_.reset();
  std::ignore = base::test::RunUntil([]() { return true; });
  provider_remote_.reset();
}

mojo::AssociatedRemote<mojom::WebNNGraphBuilder>
WebNNGraphImplFuzzTestBase::BindNewGraphBuilderRemote() {
  mojo::AssociatedRemote<mojom::WebNNGraphBuilder> remote;
  webnn_context_->CreateGraphBuilder(remote.BindNewEndpointAndPassReceiver());
  return remote;
}

struct Pool2dParams {
  OperandDataType data_type;
  Pool2dKind pool2d_kind;
  uint32_t b;
  uint32_t c;
  uint32_t ih;
  uint32_t iw;
  uint32_t wh;
  uint32_t ww;
  uint32_t b_pad_h;
  uint32_t b_pad_w;
  uint32_t e_pad_h;
  uint32_t e_pad_w;
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t dilation_h;
  uint32_t dilation_w;
  bool is_input_constant;
};

struct Conv2dParams {
  OperandDataType data_type;
  mojom::Conv2d::Kind conv2d_kind;
  uint32_t b;
  uint32_t ic;
  uint32_t ih;
  uint32_t iw;
  uint32_t oc;
  uint32_t fh;
  uint32_t fw;
  uint32_t b_pad_h;
  uint32_t b_pad_w;
  uint32_t e_pad_h;
  uint32_t e_pad_w;
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t dilation_h;
  uint32_t dilation_w;
  uint32_t groups;
  bool is_input_constant;
  bool is_filter_constant;
  bool is_bias_constant;
};

class WebNNGraphImplFuzzTest
    : public fuzztest::PerFuzzTestFixtureAdapter<WebNNGraphImplFuzzTestBase> {
 public:
  void SingleOpConv2d(Conv2dParams params, uint8_t seed_for_data);
  void SingleOpPool2d(Pool2dParams params, uint8_t seed_for_data);
};

struct BuildConv2dAttributes {
  std::vector<uint32_t> padding;
  std::vector<uint32_t> strides;
  std::vector<uint32_t> dilations;
  uint32_t groups;
};

void PrintConv2dParams(Conv2dParams params) {
  LOG(ERROR) << "Conv2dParams{data_type: " << static_cast<int>(params.data_type)
             << ", conv2d_kind: " << static_cast<int>(params.conv2d_kind)
             << ", b: " << params.b << ", ic: " << params.ic
             << ", ih: " << params.ih << ", iw: " << params.iw
             << ", oc: " << params.oc << ", fh: " << params.fh
             << ", fw: " << params.fw << ", b_pad_h: " << params.b_pad_h
             << ", b_pad_w: " << params.b_pad_w
             << ", e_pad_h: " << params.e_pad_h
             << ", e_pad_w: " << params.e_pad_w
             << ", stride_h: " << params.stride_h
             << ", stride_w: " << params.stride_w
             << ", dilation_h: " << params.dilation_h
             << ", dilation_w: " << params.dilation_w
             << ", groups: " << params.groups
             << ", is_input_constant: " << params.is_input_constant
             << ", is_filter_constant: " << params.is_filter_constant
             << ", is_bias_constant: " << params.is_bias_constant << "}";
}

void WebNNGraphImplFuzzTest::SingleOpConv2d(Conv2dParams params,
                                            uint8_t seed_for_data) {
  // PrintConv2dParams(params);

  InputOperandLayout input_layout = context_properties_.input_operand_layout;

  std::vector<uint32_t> input_dims;
  std::vector<uint32_t> filter_dims;
  if (input_layout == InputOperandLayout::kNchw) {
    input_dims = {params.b, params.ic, params.ih, params.iw};
    filter_dims = {params.oc, params.ic, params.fh, params.fw};
  } else {
    input_dims = {params.b, params.ih, params.iw, params.ic};
    filter_dims = {params.oc, params.fh, params.fw, params.ic};
  }

  if (params.ic % params.groups != 0 || params.oc % params.groups != 0) {
    params.groups = 1;
  }

  ASSIGN_OR_RETURN_VOID(auto input_desc, OperandDescriptor::Create(
                                             context_properties_,
                                             params.data_type, input_dims, ""));
  ASSIGN_OR_RETURN_VOID(
      auto filter_desc,
      OperandDescriptor::Create(context_properties_, params.data_type,
                                filter_dims, ""));
  ASSIGN_OR_RETURN_VOID(auto bias_desc, OperandDescriptor::Create(
                                            context_properties_,
                                            params.data_type, {params.oc}, ""));

  Conv2dAttributes attr;
  attr.padding.beginning = {params.b_pad_h, params.b_pad_w};
  attr.padding.ending = {params.e_pad_h, params.e_pad_w};
  attr.strides = {params.stride_h, params.stride_w};
  attr.dilations = {params.dilation_h, params.dilation_w};
  attr.groups = params.groups;
  attr.bias_operand = bias_desc;
  attr.input_layout = input_layout;
  if (input_layout == InputOperandLayout::kNhwc) {
    attr.filter_layout = Conv2dFilterOperandLayout::kOhwi;
  } else {
    attr.filter_layout = Conv2dFilterOperandLayout::kOihw;
  }

  auto output_desc_result = ValidateConv2dAndInferOutput(
      context_properties_, input_desc, filter_desc, attr);
  if (!output_desc_result.has_value()) {
    return;
  }
  auto& output_desc = output_desc_result.value();

  mojo::AssociatedRemote<mojom::WebNNGraphBuilder> remote =
      BindNewGraphBuilderRemote();
  GraphInfoBuilder builder(remote);

  OperandId input_id;
  OperandId filter_id;
  OperandId bias_id;
  std::vector<uint8_t> input_data(input_desc.PackedByteLength(), seed_for_data);
  std::vector<uint8_t> filter_data(filter_desc.PackedByteLength(),
                                   seed_for_data);
  std::vector<uint8_t> bias_data(bias_desc.PackedByteLength(), seed_for_data);

  base::flat_map<std::string, base::span<const uint8_t>> named_inputs;
  if (params.is_input_constant) {
    input_id = builder.BuildConstant(input_desc.shape(), input_desc.data_type(),
                                     base::as_byte_span(input_data));
  } else {
    input_id =
        builder.BuildInput("input", input_desc.shape(), input_desc.data_type());
    named_inputs.insert({"input", input_data});
  }
  if (params.is_filter_constant) {
    filter_id =
        builder.BuildConstant(filter_desc.shape(), filter_desc.data_type(),
                              base::as_byte_span(filter_data));
  } else {
    filter_id = builder.BuildInput("filter", filter_desc.shape(),
                                   filter_desc.data_type());
    named_inputs.insert({"filter", filter_data});
  }
  if (params.is_bias_constant) {
    bias_id = builder.BuildConstant(bias_desc.shape(), bias_desc.data_type(),
                                    base::as_byte_span(bias_data));
  } else {
    bias_id =
        builder.BuildInput("bias", bias_desc.shape(), bias_desc.data_type());
    named_inputs.insert({"bias", bias_data});
  }

  OperandId output_id = builder.BuildOutput("output", output_desc.shape(),
                                            output_desc.data_type());

  BuildConv2dAttributes conv2d_attr;
  conv2d_attr.padding = {params.b_pad_h, params.e_pad_h, params.b_pad_w,
                         params.e_pad_w};
  conv2d_attr.strides = {params.stride_h, params.stride_w};
  conv2d_attr.dilations = {params.dilation_h, params.dilation_w};
  conv2d_attr.groups = params.groups;
  builder.BuildConv2d(params.conv2d_kind, input_id, filter_id, output_id,
                      conv2d_attr, bias_id);

  if (!builder.IsValidGraphForTesting(context_properties_)) {
    return;
  }
  BuildAndCompute(context(), std::move(remote), builder.TakeGraphInfo(),
                  std::move(named_inputs));

  GetGlobalFuzzEnvironment().GetWebNNTestEnvironment().RunUntilIdle();
}

mojom::Pool2d::Kind ToMojomPool2dKind(Pool2dKind kind) {
  switch (kind) {
    case Pool2dKind::kAverage:
      return mojom::Pool2d::Kind::kAveragePool2d;
    case Pool2dKind::kL2:
      return mojom::Pool2d::Kind::kL2Pool2d;
    case Pool2dKind::kMax:
      return mojom::Pool2d::Kind::kMaxPool2d;
  }
}

struct BuildPool2dAttributes {
  std::vector<uint32_t> window_dimensions;
  std::vector<uint32_t> padding;
  std::vector<uint32_t> strides;
  std::vector<uint32_t> dilations;
};

void WebNNGraphImplFuzzTest::SingleOpPool2d(Pool2dParams params,
                                            uint8_t seed_for_data) {
  InputOperandLayout input_layout = context_properties_.input_operand_layout;

  std::vector<uint32_t> input_dims;
  if (input_layout == InputOperandLayout::kNchw) {
    input_dims = {params.b, params.c, params.ih, params.iw};
  } else {
    input_dims = {params.b, params.ih, params.iw, params.c};
  }

  ASSIGN_OR_RETURN_VOID(auto input_desc, OperandDescriptor::Create(
                                             context_properties_,
                                             params.data_type, input_dims, ""));

  Pool2dAttributes attr;
  attr.window_dimensions = Size2d<uint32_t>{.height = params.wh,
                                            .width = params.ww};
  attr.padding.beginning = {params.b_pad_h, params.b_pad_w};
  attr.padding.ending = {params.e_pad_h, params.e_pad_w};
  attr.strides = {params.stride_h, params.stride_w};
  attr.dilations = {params.dilation_h, params.dilation_w};
  attr.layout = input_layout;

  auto output_desc_result = ValidatePool2dAndInferOutput(
      context_properties_, input_desc, attr, params.pool2d_kind);
  if (!output_desc_result.has_value()) {
    return;
  }
  auto& output_desc = output_desc_result.value();

  mojo::AssociatedRemote<mojom::WebNNGraphBuilder> remote =
      BindNewGraphBuilderRemote();
  GraphInfoBuilder builder(remote);

  OperandId input_id;
  std::vector<uint8_t> input_data(input_desc.PackedByteLength(), seed_for_data);

  base::flat_map<std::string, base::span<const uint8_t>> named_inputs;
  if (params.is_input_constant) {
    input_id = builder.BuildConstant(input_desc.shape(), input_desc.data_type(),
                                     base::as_byte_span(input_data));
  } else {
    input_id =
        builder.BuildInput("input", input_desc.shape(), input_desc.data_type());
    named_inputs.insert({"input", input_data});
  }

  OperandId output_id = builder.BuildOutput("output", output_desc.shape(),
                                            output_desc.data_type());

  BuildPool2dAttributes pool2d_attr;
  pool2d_attr.window_dimensions = {params.wh, params.ww};
  pool2d_attr.padding = {params.b_pad_h, params.e_pad_h, params.b_pad_w,
                         params.e_pad_w};
  pool2d_attr.strides = {params.stride_h, params.stride_w};
  pool2d_attr.dilations = {params.dilation_h, params.dilation_w};
  builder.BuildPool2d(ToMojomPool2dKind(params.pool2d_kind), input_id,
                      output_id, pool2d_attr);

  if (!builder.IsValidGraphForTesting(context_properties_)) {
    return;
  }
  BuildAndCompute(context(), std::move(remote), builder.TakeGraphInfo(),
                  std::move(named_inputs));

  GetGlobalFuzzEnvironment().GetWebNNTestEnvironment().RunUntilIdle();
}

auto AnyOperandDataType() {
  return fuzztest::ElementOf<OperandDataType>(
      {OperandDataType::kFloat32, OperandDataType::kFloat16,
       OperandDataType::kInt32, OperandDataType::kUint32,
       OperandDataType::kInt64, OperandDataType::kUint64,
       OperandDataType::kInt8, OperandDataType::kUint8, OperandDataType::kUint4,
       OperandDataType::kInt4});
}

auto AnyConv2dKind() {
  return fuzztest::ElementOf<mojom::Conv2d::Kind>(
      {mojom::Conv2d::Kind::kDirect, mojom::Conv2d::Kind::kTransposed});
}

auto AnyConv2dParams() {
  return fuzztest::StructOf<Conv2dParams>(
      AnyOperandDataType(), AnyConv2dKind(),
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // b
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // ic
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // ih
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // iw
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // oc
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // fh
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // fw
      fuzztest::InRange<uint32_t>(0, std::numeric_limits<int32_t>::max()),  // b_pad_h
      fuzztest::InRange<uint32_t>(0, std::numeric_limits<int32_t>::max()),  // b_pad_w
      fuzztest::InRange<uint32_t>(0, std::numeric_limits<int32_t>::max()),  // e_pad_h
      fuzztest::InRange<uint32_t>(0, std::numeric_limits<int32_t>::max()),  // e_pad_w
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int16_t>::max()),  // stride_h
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int16_t>::max()),  // stride_w
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int16_t>::max()),  // dilation_h
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int16_t>::max()),  // dilation_w
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // groups
      fuzztest::Arbitrary<bool>(),            // is_input_constant
      fuzztest::Arbitrary<bool>(),            // is_filter_constant
      fuzztest::Arbitrary<bool>()             // is_bias_constant
  );
}

FUZZ_TEST_F(WebNNGraphImplFuzzTest, SingleOpConv2d)
    .WithDomains(AnyConv2dParams(), fuzztest::Arbitrary<uint8_t>())
    .WithSeeds({{{OperandDataType::kFloat16,
                  mojom::Conv2d::Kind::kDirect,
                  1,
                  3,
                  224,
                  224,
                  64,
                  7,
                  7,
                  3,
                  3,
                  3,
                  3,
                  1,
                  1,
                  1,
                  1,
                  1,
                  false,
                  true,
                  true},
                 1},
                // {{OperandDataType::kFloat16,
                //   mojom::Conv2d::Kind::kDirect,
                //   1,
                //   3,
                //   224,
                //   224,
                //   64,
                //   7,
                //   7,
                //   3,
                //   268435459,
                //   3,
                //   3,
                //   1,
                //   6040,
                //   1,
                //   1,
                //   1,
                //   false,
                //   true,
                //   false},
                //  1},
                });

auto AnyPool2dKind() {
  return fuzztest::ElementOf<Pool2dKind>(
      {Pool2dKind::kAverage, Pool2dKind::kL2, Pool2dKind::kMax});
}

auto AnyPool2dParams() {
  return fuzztest::StructOf<Pool2dParams>(
      AnyOperandDataType(), AnyPool2dKind(),
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // b
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // c
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // ih
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // iw
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // wh
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int32_t>::max()),  // ww
      fuzztest::InRange<uint32_t>(0, std::numeric_limits<int32_t>::max()),  // b_pad_h
      fuzztest::InRange<uint32_t>(0, std::numeric_limits<int32_t>::max()),  // b_pad_w
      fuzztest::InRange<uint32_t>(0, std::numeric_limits<int32_t>::max()),  // e_pad_h
      fuzztest::InRange<uint32_t>(0, std::numeric_limits<int32_t>::max()),  // e_pad_w
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int16_t>::max()),  // stride_h
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int16_t>::max()),  // stride_w
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int16_t>::max()),  // dilation_h
      fuzztest::InRange<uint32_t>(1, std::numeric_limits<int16_t>::max()),  // dilation_w
      fuzztest::Arbitrary<bool>()             // is_input_constant
  );
}

FUZZ_TEST_F(WebNNGraphImplFuzzTest, SingleOpPool2d)
    .WithDomains(AnyPool2dParams(), fuzztest::Arbitrary<uint8_t>())
    .WithSeeds({{{OperandDataType::kFloat32,
                  Pool2dKind::kMax,
                  1,
                  3,
                  4,
                  4,
                  2,
                  2,
                  0,
                  0,
                  0,
                  0,
                  2,
                  2,
                  1,
                  1,
                  false},
                 1}});

}  // namespace webnn::test
