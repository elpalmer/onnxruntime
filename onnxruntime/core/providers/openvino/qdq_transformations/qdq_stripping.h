// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once


#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace openvino_ep {

// Creates a new model without the DQ/Q operators in the src graph as per pre-defined rulesets
Status CreateModelWithStrippedQDQNodes(const GraphViewer& src_graph,
                          const logging::Logger& logger,
                          int32_t float_type,
                          /*out*/ std::unique_ptr<onnxruntime::Model>& model);

}
}