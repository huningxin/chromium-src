// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/mojo_client.h"

#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"

namespace blink {

MojoClient::MojoClient(ExecutionContext* execution_context)
    : mojo_server_(execution_context) {
  execution_context->GetBrowserInterfaceBroker().GetInterface(
      mojo_server_.BindNewPipeAndPassReceiver(
          execution_context->GetTaskRunner(TaskType::kInternalDefault)));
}

void MojoClient::CreateMojoContext(ScriptPromiseResolver* resolver,
                                   ContextOptionsPtr options,
                                   MojoServer::CreateContextCallback callback) {
  if (!mojo_server_.is_bound()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError, "WebNN server unavailable."));
    return;
  }
  mojo_server_->CreateContext(std::move(options), std::move(callback));
}

void MojoClient::CreateMojoContextSync(
    ContextOptionsPtr options,
    ::mojo::PendingRemote<::ml::webnn::mojom::blink::Context>* pending_remote,
    ExceptionState& exception_state) {
  if (!mojo_server_.is_bound()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                      "WebNN server unavailable.");
    return;
  }
  if (!mojo_server_->CreateContext(std::move(options), pending_remote)) {
    exception_state.ThrowDOMException(DOMExceptionCode::kUnknownError,
                                      "Failed to create the context.");
  };
}

void MojoClient::Trace(Visitor* visitor) const {
  visitor->Trace(mojo_server_);
}

}  // namespace blink
