#pragma once

#include <vector>
#include <string>
#include "core/providers/openvino/openvino_execution_provider.h"

// #include "onnxruntime_c_api.h"

// #ifdef __cplusplus

// namespace onnxruntime{
  struct OpenVINO_Devices {
    // std::vector<std::string> GetAvailableDevices_ov();
    std::vector<std::string> GetAvailableDevices_ov() {
        onnxruntime::openvino_ep::OVCore ie_core;
        return ie_core.GetAvailableDevices();
}

  };
// }

// #endif
