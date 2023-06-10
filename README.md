# Boson

A Vulkan abstraction layer that makes graphics easy and enjoyable.

## Design Philosophy

### Simple yet efficient.

This library is meant to simplify Vulkan without sacrificing usability.

### Built-ins

In addition, it is meant to contain common patterns and practices that level up the graphics programming experience (such as bindless resources).

### Performant

It is also meant to be a thin layer above Vulkan, thus not sacrificing any performance.

### Move fast, break things

This library is under constant development so that it can meet any requirements. Thus, backwards compatibility is not yet a concern.

## Features

- [x] Simplified API surface: No more 1000 LOC to draw a triangle (takes about 300-500).
- [x] Shader hot reloading.
- [x] Bindless shader resource model.
- [x] Render graph implementation: featuring resource tracking and automatic synchronization.
- [ ] Render graph optimization: There is still more work to do to make this as fast as possible.
- [ ] Full Vulkan support: I want this library to allow you to do anything Vulkan offers, so if something is not present, submit an issue or PR.

## Your first triangle (click the image!)

[![Watch the video](https://img.youtube.com/vi/BlCp4RUuC4I/maxresdefault.jpg)](https://youtu.be/BlCp4RUuC4I)

## Todo / Ideas for improvement

Have an idea? Let me know through an issue or PR.

- Create documentation.
- Create examples.
- Fix all warnings.
- Add debug naming of resources for RenderDoc.
- Conditional render graph tasks.
- Built-in error handling. (no more having to use Vulkan Configuration / debug messaging)
- Add all formats.
- Add hardware raytracing support.
- Better glslang integration. (stop using Vulkan SDK through shell process and instead integrate through bindings)
- Custom compiler traits and more shader language support? (HLSL)
- Remove pub(crate) items from the prelude.
