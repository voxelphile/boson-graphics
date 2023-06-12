use std::fs;

//Comes with a prelude so you do not have to import individual types.
use boson::prelude::*;

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

fn main() {
    //Initialize winit like any other application.
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    //The context is your gateway to boson. It starts the vulkan backend and roughly translates to the vulkan instance.
    let context = Context::new(ContextInfo {
        //This tells the backend that you would like to print debug information.
        //Useful for finding a bug in your code.
        //There is performance loss.
        //Use Vulkan Configurator (comes with Vulkan SDK) for more advanced use cases.
        enable_validation: false,
        //Give your application a name
        application_name: "Triangle",
        //Almost everything in this library has a default.
        ..Default::default()
    })
    .expect("failed to create context");

    //The device is your handle to the system GPU.
    //It automatically selects the most capable GPU. (this can be configured)
    let mut device = context
        .create_device(DeviceInfo {
            //The following two parameters use the raw_window_handle crate.
            //Compatible with winit or your favorite windowing system.
            display: window.raw_display_handle(),
            window: window.raw_window_handle(),
            ..Default::default()
        })
        .expect("failed to create device");

    //The swapchain manages the images which are sent to the monitor to be displayed.
    //You must use the device to acquire images from the swapchain in your render graph.
    //We put this inside a function because we want to change it when its resized.
    fn resize(
        device: Device,
        old_swapchain: Option<Swapchain>,
        width: u32,
        height: u32,
    ) -> (Swapchain, RenderGraph<'static, RenderInfo>) {
        let mut swapchain = device
            .create_swapchain(SwapchainInfo {
                width,
                height,
                present_mode: PresentMode::DoNotWaitForVBlank,
                image_usage: ImageUsage::COLOR,
                ..Default::default()
            })
            .expect("failed to create swapchain");

        //We will get to this later.
        let render_graph = record(device.clone(), swapchain, width, height);

        (swapchain, render_graph)
    }

    let mut swapchain = None;
    //We will get to this later.
    let mut render_graph = None;

    //Initial resize
    {
        let winit::dpi::PhysicalSize { width, height } = window.inner_size();
        //The device is a handle so you can clone it whenever.
        let (s, r) = resize(device.clone(), swapchain, width, height);
        swapchain = Some(s);
        render_graph = Some(r);
    }

    //A pipeline compiler allows us to load and hot-reload pipelines.
    //This used to be a file manager too, however that has changed.
    //By default, the include directory is your working directory. This (during development) defaults to where your Cargo.toml lives.
    let pipeline_compiler = device.create_pipeline_compiler(Default::default());

    //This used to have annoying lifetime issues and the like.
    //However it is possible to now clone the pipeline.
    //Have fun with this! ^^
    let render_pipeline = pipeline_compiler
        .create_graphics_pipeline(GraphicsPipelineInfo {
            //By default, boson will inject `#define the_shader_type_in_lowercase`
            //for example, the first shader will have `#define vertex` added to the top of the file before compilation
            //and the second shader will have `#define fragment`
            shaders: vec![
                Shader{
                    ty: ShaderType::Vertex,
                    //Here we read the source file into a string. We provide this to the pipeline compiler whenever we want to load or hot reload.
                    source: fs::read_to_string("./triangle.glsl").unwrap(),
                    defines: vec![]
                },
                Shader{
                    ty: ShaderType::Fragment,
                    //The fragment shader is in the same file.
                    source: fs::read_to_string("./triangle.glsl").unwrap(),
                    defines: vec![]
                },
            ],
            color: vec![Color {
                format: device
                    .presentation_format(
                        swapchain.expect("should have been set in the initial resize"),
                    )
                    .expect("this shouldnt fail"),
                ..Default::default()
            }],
            ..Default::default()
        })
        .expect("failed to create pipeline");

    //TODO: this explanation can be improved.
    //A semaphore facilities synchronization between different parts of the GPU pipeline.
    //In this example, we signal the `acquire_semaphore` when we get an image from the swapchain.
    //Then, we wait for this signal to be processed by the GPU before we signal the `present_semaphore`.
    //This allows the gpu to process multiple frames at a time without tripping on itself.
    //Note: These are not necessary if you are making a rendergraph that is only executed once.
    let acquire_semaphore = device
        .create_binary_semaphore(Default::default())
        .expect("failed to create semaphore");
    let present_semaphore = device
        .create_binary_semaphore(Default::default())
        .expect("failed to create semaphore");

    //Data as your seeing it here lives on the CPU. To move data from the CPU to the GPU,
    //we need a buffer that lives on the cpu and can move data to the gpu.
    //This is called a staging buffer, usually, which is a buffer with host access.
    let staging_buffer = device
        .create_buffer(BufferInfo {
            size: 1024,
            //this tells boson we would like to be able to write/read to/from this buffer from the cpu.
            memory: Memory::HOST_ACCESS,
            ..Default::default()
        })
        .expect("failed to create buffer");

    //To demonstrate usage of resources, we will create a buffer that holds one value:
    //the amount of time that has passed since the start of execution,
    //such that we can move the triangle based upon time.
    let time_buffer = device
        .create_buffer(BufferInfo {
            size: 1024,
            //we do not need any special parameters here.
            ..Default::default()
        })
        .expect("failed to create buffer");

    //You must pass in a struct that contains the information you will use in the render graph.
    //In `bevy` for example, I use the Entity-Component-System `World`.
    //However, you can use any struct.
    struct RenderInfo {
        //We will need a clone of the device handle to acquire the next swapchain image,
        //as well as the swapchain handle.
        device: Device,
        swapchain: Swapchain,
        //We will also need access to the semaphores for proper synchronization.
        acquire_semaphore: BinarySemaphore,
        present_semaphore: BinarySemaphore,
        //These are the buffers we previously defined.
        staging_buffer: Buffer,
        time_buffer: Buffer,
        //This is the render pipeline we previously defined
        //TODO: I want to remove the lifetime and const generics.
        render_pipeline: Pipeline,
        //This is what you will send to the shader, either through a buffer or push constant,
        //that lets the shader know where in memory your buffer is.
        time_buffer_address: BufferAddress,
        //This is the value we will upload to the gpu per frame.
        time: f32,
    };

    fn record(
        device: Device,
        swapchain: Swapchain,
        width: u32,
        height: u32,
    ) -> RenderGraph<'static, RenderInfo> {
        //Now here comes the fun part. We use the device to start building a render graph.
        //A render graph is just what it sounds like: a graph.
        //The nodes of this graph are the work that you do, and the edges are determined
        //based upon how you use the resources in your application.
        //You need access to the swapchain regardless of whether you use this render graph for presentation to the screen.
        //(this may be removed in the future)
        let mut render_graph_builder = device
            .create_render_graph::<'_, RenderInfo>(RenderGraphInfo {
                swapchain,
                ..Default::default()
            })
            .expect("failed to create render graph builder");

        //A task is a unit of work for the GPU. Synchronization happens between tasks.
        //Things inside a task may be reordered unless explicitly specified not to through the use of `commands.pipeline_barrier(..)`
        //Underneath the hood, the render graph tracks resource usage and calls `commands.pipeline_barrier(..)` for you.
        render_graph_builder.add(Task {
            resources: vec![Resource::Buffer(
                //This closure tells the render graph what resource you are using.
                //It is polled every frame.
                Box::new(|render_info| render_info.staging_buffer),
                //This tells the render graph how you use this resource.
                //It is very important this is accurate to your usecase or you may get a crash or undefined behavior.
                BufferAccess::HostTransferWrite,
            )],
            task: |render_info, commands| {
                commands.write_buffer(BufferWrite {
                    //This is the index of the buffer in the resources vector.
                    //If you have multiple buffers, images, then you will want to make sure this is accurate.
                    buffer: 0,
                    //This is where you are writing in the buffer
                    offset: 0,
                    //This is a slice that is copied bit-for-bit into the buffer.
                    src: &[render_info.time],
                })
            },
        });

        //This should start being self-explanatory.
        render_graph_builder.add(Task {
            resources: vec![
                //Now we copy from this buffer.
                Resource::Buffer(
                    Box::new(|render_info| render_info.staging_buffer),
                    BufferAccess::TransferWrite,
                ),
                //To this buffer
                Resource::Buffer(
                    Box::new(|render_info| render_info.time_buffer),
                    BufferAccess::TransferRead,
                ),
            ],
            task: |render_info, commands| {
                commands.copy_buffer_to_buffer(BufferCopy {
                    //Index of the buffer in the resource vector
                    from: 0,
                    //Index of the buffer in the resource vector
                    to: 1,
                    //Source (from parameter) buffer offset.
                    src: 0,
                    //Destination (to parameter) buffer offset.
                    dst: 0,
                    size: std::mem::size_of::<f32>(),
                })
            },
        });

        //Here comes the super fun part. Drawing to the screen.
        render_graph_builder.add(Task {
            resources: vec![
                Resource::Image(
                    //This acquires the next image from the swapchain, essentially the image we can draw to
                    //and it is also the image which is drawn to the screen.
                    Box::new(|render_info| {
                        render_info
                            .device
                            .acquire_next_image(Acquire {
                                swapchain: render_info.swapchain,
                                semaphore: Some(render_info.acquire_semaphore),
                            })
                            .expect("failed to acquire next image")
                    }),
                    ImageAccess::ColorAttachment,
                    //This is the image aspect. It defaults to color images.
                    //If you are using a depth image, you will want `ImageAspect::DEPTH`
                    //and if you use the stencil buffer you will want to add `ImageAspect::STENCIL`
                    Default::default(),
                ),
                Resource::Buffer(
                    Box::new(|render_info| render_info.time_buffer),
                    BufferAccess::VertexShaderReadOnly,
                ),
            ],
            //We specify the closure as `move` so that it can take the values of `width` and `height`
            //See `commands.start_rendering` and `commands.set_resolution`
            task: move |render_info, commands| {
                //No need for renderpasses here! just start rendering.
                commands.start_rendering(Render {
                    color: vec![Attachment {
                        //This is the index into the resource vector.
                        image: 0,
                        //What you want to do with the image to start.
                        //In this case, we are setting it to `clear`...
                        load_op: LoadOp::Clear,
                        //...which is black.
                        clear: Clear::Color(0.0, 0.0, 0.0, 1.0),
                    }],
                    depth: None,
                    use_stencil: false,
                    render_area: RenderArea {
                        x: 0,
                        y: 0,
                        width,
                        height,
                    },
                })?;
                //^ There is (supposed) to be proper error handling, although its not all implemented yet.
                //So add a question mark to the end of your statement like this.

                commands.set_resolution(
                    (width, height),
                    /*flip the viewport on the y axis, if your coming from opengl, you will maybe want this*/ 
                    true,
                );

                //A push constant is like data that hitches a ride with the command to the GPU.
                //Make sure this is #[repr(C)]!
                //Also needs to know the pipeline to set the push constant for.
                #[repr(C)]
                #[derive(Clone, Copy)]
                struct Push {
                    time_buffer_address: BufferAddress,
                }

                commands.push_constant(PushConstant {
                    data: Push {
                        time_buffer_address: render_info.time_buffer_address,
                    },
                    pipeline: &render_info.render_pipeline,
                })?;

                //Finally, before rendering, specify the pipeline you will use.
                commands.set_pipeline(&render_info.render_pipeline)?;

                commands.draw(Draw {
                    //Three vertices in a triangle.
                    vertex_count: 3,
                })?;

                commands.end_rendering()
            },
        });

        render_graph_builder.add(Task {
            resources: vec![Resource::Image(
                //This will return the same image within the same frame. When you go to the next frame,
                //it will send the next image. and so on and so forth.
                Box::new(|render_info| {
                    render_info
                        .device
                        .acquire_next_image(Acquire {
                            swapchain: render_info.swapchain,
                            semaphore: Some(render_info.acquire_semaphore),
                        })
                        .expect("failed to acquire next image")
                }),
                ImageAccess::ColorAttachment,
                Default::default(),
            )],
            task: |render_info, commands| {
                //This tells the render graph to send the tasks to the GPU.
                //Without it, nothing will happen.
                commands.submit(Submit {
                    wait_semaphore: Some(render_info.acquire_semaphore),
                    signal_semaphore: Some(render_info.present_semaphore),
                })?;

                //This tells the GPU to show what we drew to the screen.
                commands.present(Present {
                    wait_semaphore: render_info.present_semaphore,
                })?;
                Ok(())
            },
        });

        render_graph_builder
            .complete()
            .expect("failed to create render graph")
    }

    let mut render_info = RenderInfo {
        device: device.clone(),
        swapchain: swapchain.unwrap(),
        acquire_semaphore,
        present_semaphore,
        render_pipeline,
        staging_buffer,
        time_buffer,
        //This is how you get the shader address of a buffer.
        time_buffer_address: device.address(time_buffer).unwrap(),
        time: 0.0,
    };

    let start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                println!("The close button was pressed; stopping");
                control_flow.set_exit();
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                render_info.time = std::time::Instant::now()
                    .duration_since(start_time)
                    .as_secs_f32();
                render_graph.as_mut().unwrap().render(&mut render_info);
            }
            _ => (),
        }
    });
}
