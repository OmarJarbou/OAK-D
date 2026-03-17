# C++ Camera Stream

## Overview

This example showcases the smallest possible DepthAIv3 pipeline written in C++ that streams raw color frames to the DepthAI Visualizer. It creates a single `dai::node::Camera`, requests an `1280x800` NV12 output from `CAM_A`, registers that output on a `dai::RemoteConnection`, and keeps the pipeline alive until you quit from the Visualizer. Use it as a template when you need a C++ starting point for experimenting with camera streaming or registering your own topics.

## Demo

![Camera stream demo](./media/camera_stream.gif)

The application renders directly inside the DepthAI Visualizer. After you start it (see the steps below), open [http://localhost:8082](http://localhost:8082) and look for the `images` stream tile; focusing the browser tab lets you send hotkeys (including `q`) back to the app.

## Peripheral Mode

### Installation

Prepare a development environment with:

- [DepthAI C++ library (`depthai-core`)](https://github.com/luxonis/depthai-core) installed and discoverable.
- CMake **3.20+**.
- A C++17 capable compiler

To install the DepthAI C++ library, follow the instructions [here](https://github.com/luxonis/depthai-core?tab=readme-ov-file#installation-and-integration).
In short:

```bash
git clone --recurse-submodules https://github.com/luxonis/depthai-core.git
cd depthai-core
mkdir build
mkdir -p /opt/Luxonis/depthai-core
cmake -S . -B build -DBUILD_SHARED_LIBS=ON # Optionally use a custom installation path here like -DCMAKE_INSTALL_PREFIX=/tmp/depthai-core
cmake --build build --parallel
sudo cmake --install build # Sudo is not needed if you used a custom installation path above
```

### Build & Run

```bash
cd cpp/camera_stream
cmake -S . -B build # In case you used CMAKE_INSTALL_PREFIX above, add -DCMAKE_PREFIX_PATH=/tmp/depthai-core
cmake --build build
./build/main
```

Keep the terminal open—the process stays alive while the pipeline runs. Open [http://localhost:8082](http://localhost:8082) in your browser to view the live `images` stream, and press `q` to stop the application cleanly.

## Standalone Mode (RVC4 only)

Running the example in standalone mode builds and deploys it as an OAK app so that it runs completely on the device.

1. Install `oakctl` by following the instructions [here](https://docs.luxonis.com/software-v3/oak-apps/oakctl).

2. Connect to your device:

   ```bash
   oakctl connect <DEVICE_IP>
   ```

3. From the `cpp/camera_stream` directory, run the packaged app:

   ```bash
   oakctl app run .
   ```

`oakctl` uses the provided `oakapp.toml` to build the C++ project inside the Luxonis base container and deploy it to the device. Configuration tweaks such as changing the camera resolution or registering more topics should be done in `src/main.cpp`, then re-run `oakctl app run .`.
