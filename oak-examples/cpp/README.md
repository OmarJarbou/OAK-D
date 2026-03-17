# C++ Examples Overview

This section contains DepthAIv3 sample applications implemented in C++. They demonstrate how to build and deploy OAK apps with CMake-based workflows, showing how to link against `depthai-core`, configure `oakapp.toml`, and run the binaries either from a host PC or directly on RVC4 devices via `oakctl`.

## Platform Compatibility

| Name                            | RVC2 | RVC4 (peripheral) | RVC4 (standalone) | DepthAIv2 | Notes                                                                                      |
| ------------------------------- | ---- | ----------------- | ----------------- | --------- | ------------------------------------------------------------------------------------------ |
| [camera_stream](camera_stream/) | ✅   | ✅                | ✅                |           | Minimal C++ pipeline streaming camera frames through `dai::RemoteConnection` to Visualizer |
| [UVC example](uvc/)             | ❌   | ❌                | ✅                |           | C++ example for streaming video via UVC (USB Video Class) (device behaves as UVC camera)   |

✅: available; ❌: not available; 🚧: work in progress
