// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/mojo_server_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/dml/context_dml_impl.h"
#include "content/browser/ml/webnn/dml/webnn_service_dml_impl.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

// HACK:::
#pragma optimize("", off)

namespace content::webnn {

namespace {

using ml::webnn::mojom::Context;
using ml::webnn::mojom::ContextOptionsPtr;
using ml::webnn::mojom::MojoServer;

}  // namespace

// static
void MojoServerDMLImpl::Create(mojo::PendingReceiver<MojoServer> receiver,
                               WebnnServiceDMLImpl* webnn_service) {
  mojo::MakeSelfOwnedReceiver<MojoServer>(
      base::WrapUnique(new MojoServerDMLImpl(webnn_service)),
      std::move(receiver));
}

MojoServerDMLImpl::~MojoServerDMLImpl() = default;

MojoServerDMLImpl::MojoServerDMLImpl(WebnnServiceDMLImpl* webnn_service)
    : webnn_service_(webnn_service) {}

void MojoServerDMLImpl::CreateContext(
    ContextOptionsPtr options,
    MojoServer::CreateContextCallback callback) {
  auto adapter = webnn_service_->RequestAdapter(options->power_preference);
  if (!adapter) {
    std::move(callback).Run(mojo::NullRemote());
    return;
  }

  auto context = std::make_unique<ContextDMLImpl>(adapter);
  HRESULT hr = context->Initialize();
  if (FAILED(hr)) {
    DLOG(ERROR) << "Initialize context failed: "
                << logging::SystemErrorCodeToString(hr);
    std::move(callback).Run(mojo::NullRemote());
    return;
  }

  // The remote sent to the renderer.
  mojo::PendingRemote<Context> blink_remote;
  // The receiver bind to ContextDMLImpl.
  mojo::MakeSelfOwnedReceiver<Context>(
      base::WrapUnique(context.release()),
      blink_remote.InitWithNewPipeAndPassReceiver());

  std::move(callback).Run(std::move(blink_remote));
}

bool MojoServerDMLImpl::CreateContext(
    ContextOptionsPtr options,
    ::mojo::PendingRemote<::ml::webnn::mojom::Context>* out_remote) {
  auto adapter = webnn_service_->RequestAdapter(options->power_preference);
  if (!adapter) {
    DLOG(ERROR) << "Failed to request the adapter.";
    return false;
  }

  auto context = std::make_unique<ContextDMLImpl>(adapter);
  HRESULT hr = context->Initialize();
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to initialize the context: "
                << logging::SystemErrorCodeToString(hr);
    return false;
  }

  // The receiver bind to ContextDMLImpl.
  mojo::MakeSelfOwnedReceiver<Context>(
      base::WrapUnique(context.release()),
      out_remote->InitWithNewPipeAndPassReceiver());

  return true;
}

}  // namespace content::webnn
