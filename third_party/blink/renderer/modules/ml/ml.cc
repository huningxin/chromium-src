// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml.h"

#include "build/buildflag.h"
#include "components/ml/mojom/web_platform_model.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_client.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

#pragma optimize("", off) // TODO:::DELETE

namespace blink {

namespace {

using ml::model_loader::mojom::blink::CreateModelLoaderOptionsPtr;
using ml::model_loader::mojom::blink::MLService;

}  // namespace

ML::ML(ExecutionContext* execution_context)
    : ExecutionContextClient(execution_context),
      remote_service_(execution_context) {}

void ML::CreateModelLoader(ScriptState* script_state,
                           ExceptionState& exception_state,
                           CreateModelLoaderOptionsPtr options,
                           MLService::CreateModelLoaderCallback callback) {
  if (!BootstrapMojoConnectionIfNeeded(script_state, exception_state)) {
    // An exception has already been thrown in
    // `BootstrapMojoConnectionIfNeeded()`.
    return;
  }
  remote_service_->CreateModelLoader(std::move(options), std::move(callback));
}

void ML::CreateWebnnMojoContext(ScriptPromiseResolver* resolver,
                                ContextOptionsPtr options,
                                MojoServer::CreateContextCallback callback) {
  if (!webnn_mojo_client_) {
    webnn_mojo_client_ =
        MakeGarbageCollected<MojoClient>(this->GetExecutionContext());
  }

  webnn_mojo_client_->CreateMojoContext(resolver, std::move(options),
                                        std::move(callback));
}

void ML::CreateWebnnMojoContextSync(
    ContextOptionsPtr options,
    ::mojo::PendingRemote<::ml::webnn::mojom::blink::Context>* pending_remote,
    ExceptionState& exception_state) {
  if (!webnn_mojo_client_) {
    webnn_mojo_client_ =
        MakeGarbageCollected<MojoClient>(this->GetExecutionContext());
  }
  webnn_mojo_client_->CreateMojoContextSync(std::move(options), pending_remote,
                                            exception_state);
}

void ML::Trace(Visitor* visitor) const {
  visitor->Trace(remote_service_);
  ExecutionContextClient::Trace(visitor);
  visitor->Trace(webnn_mojo_client_);
  ScriptWrappable::Trace(visitor);
}

ScriptPromise ML::createContext(ScriptState* script_state,
                                MLContextOptions* option,
                                ExceptionState& exception_state) {
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid script state");
    return ScriptPromise();
  }

  ScriptPromiseResolver* resolver =
      MakeGarbageCollected<ScriptPromiseResolver>(script_state);

  auto* ml_context = MakeGarbageCollected<MLContext>(
      option->devicePreference(), option->powerPreference(),
      option->modelFormat(), option->numThreads(), this);
  if (ml_context->IsWebnnMojoContextEnabled()) {
    // WebNN mojo context need to be created in server side to map different
    // hardware acceleration and manage the processes of graph execution, so the
    // ml context will await the callback from server side and then resolve.
    ml_context->CreateWebnnMojoContext(resolver);
  } else {
    resolver->Resolve(ml_context);
  }
  return resolver->Promise();
}

MLContext* ML::createContextSync(ScriptState* script_state,
                                 MLContextOptions* options,
                                 ExceptionState& exception_state) {
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid script state");
    return nullptr;
  }

  auto* ml_context = MakeGarbageCollected<MLContext>(
      options->devicePreference(), options->powerPreference(),
      options->modelFormat(), options->numThreads(), this);
  if (ml_context->IsWebnnMojoContextEnabled()) {
    // WebNN mojo context need to be created in server side to map different
    // hardware acceleration and manage the processes of graph execution, we use
    // synchronous calls to wait for the server side.
    ml_context->CreateWebnnMojoContextSync(script_state, exception_state);
  }
  return ml_context;
}

bool ML::BootstrapMojoConnectionIfNeeded(ScriptState* script_state,
                                         ExceptionState& exception_state) {
  // We need to do the following check because the execution context of this
  // navigator may be invalid (e.g. the frame is detached).
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "The execution context is invalid");
    return false;
  }
  // Note that we do not use `ExecutionContext::From(script_state)` because
  // the ScriptState passed in may not be guaranteed to match the execution
  // context associated with this navigator, especially with
  // cross-browsing-context calls.
  if (!remote_service_.is_bound()) {
    GetExecutionContext()->GetBrowserInterfaceBroker().GetInterface(
        remote_service_.BindNewPipeAndPassReceiver(
            GetExecutionContext()->GetTaskRunner(TaskType::kInternalDefault)));
  }
  return true;
}

}  // namespace blink
