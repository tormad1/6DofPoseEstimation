# 6DoF Pose Estimation – Unity Plugin

A Unity plugin that estimates the position and rotation of a physical object using camera input. Built on GigaPose as part of a group project at Edinburgh Napier University.

## Requirements

- Unity 6.3 LTS (6000.3.8f1)
- Windows x86_64
- The plugin ships with a dummy pose backend for testing. The real inference backend requires the GigaPose setup described in `docs/template_generation.md`.

## Installation

1. Clone or download this repository.
2. Open your Unity project's `Packages/manifest.json`.
3. Add the following line to the `dependencies` block:

```json
"com.napier.sixdofposeestimation": "file:/path/to/com.napier.sixdofposeestimation"
```

1. Save the file. Unity will import the package automatically.

## Usage

1. Add `PoseManager` to a GameObject in your scene.
2. Assign a target Transform. This is the object whose position and rotation will be updated.
3. Add `PoseBridge` to the same GameObject.
4. In `PoseBridge`, set `useDummy` to `true` to test with the dummy backend.

## Components

**PoseBridge** handles communication with the native backend. It exposes `GetPose(out Pose pose)` which returns the latest pose from the backend along with a confidence score and timestamp.

**PoseManager** consumes poses from the bridge and applies them to the target Transform. It includes jump detection and tracking timeout logic to filter out bad estimates.

## Project Structure

```
com.napier.sixdofposeestimation/
├── Runtime/          # C# scripts
├── Plugins/          # Native DLLs
└── Samples~/         # Example scene
```
