use crate::device::DeviceInner;
use crate::pipeline::{PipelineInner, PipelineModify};
use crate::prelude::*;
use crate::renderpass::{Framebuffer, RenderPass};

use std::collections::HashMap;
use std::mem;
use std::ops;
use std::ptr;
use std::slice;
use std::sync::Mutex;

use ash::vk::{self, Offset2D, Rect2D, Offset3D};

use bitflags::bitflags;

pub struct Commands<'a> {
    pub(crate) device: &'a DeviceInner,
    pub(crate) qualifiers: &'a [Qualifier],
    pub(crate) swapchain: &'a Swapchain,
    pub(crate) command_buffer: &'a vk::CommandBuffer,
    pub(crate) submit: &'a mut Option<Submit>,
    pub(crate) present: &'a mut Option<Present>,
}
bitflags! {
    pub struct Access: u32 {
        const WRITE = 0x00000001;
        const READ = 0x00000002;
    }
}

fn stage_access_to_access((stage, access): (PipelineStage, Access)) -> vk::AccessFlags {
    let mut result = vk::AccessFlags::empty();

    if access.contains(Access::READ) {
        if stage.contains(PipelineStage::TOP_OF_PIPE) {
            result |= vk::AccessFlags::MEMORY_READ;
        }

        if stage.contains(PipelineStage::VERTEX_SHADER) {
            result |= vk::AccessFlags::SHADER_READ;
        }

        if stage.contains(PipelineStage::FRAGMENT_SHADER) {
            result |= vk::AccessFlags::SHADER_READ;
        }

        if stage.contains(PipelineStage::EARLY_FRAGMENT_TESTS) {
            result |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
        }

        if stage.contains(PipelineStage::LATE_FRAGMENT_TESTS) {
            result |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
        }

        if stage.contains(PipelineStage::COLOR_ATTACHMENT_OUTPUT) {
            result |= vk::AccessFlags::COLOR_ATTACHMENT_READ;
        }

        if stage.contains(PipelineStage::COMPUTE_SHADER) {
            result |= vk::AccessFlags::SHADER_READ;
        }

        if stage.contains(PipelineStage::TRANSFER) {
            result |= vk::AccessFlags::TRANSFER_READ;
        }

        if stage.contains(PipelineStage::BOTTOM_OF_PIPE) {
            result |= vk::AccessFlags::MEMORY_READ;
        }

        if stage.contains(PipelineStage::HOST) {
            result |= vk::AccessFlags::HOST_READ;
        }

        if stage.contains(PipelineStage::ALL_GRAPHICS) {
            result |= vk::AccessFlags::MEMORY_READ;
        }

        if stage.contains(PipelineStage::ALL_COMMANDS) {
            result |= vk::AccessFlags::MEMORY_READ;
        }
    }

    if access.contains(Access::WRITE) {
        if stage.contains(PipelineStage::TOP_OF_PIPE) {
            result |= vk::AccessFlags::MEMORY_WRITE;
        }

        if stage.contains(PipelineStage::VERTEX_SHADER) {
            result |= vk::AccessFlags::SHADER_WRITE;
        }

        if stage.contains(PipelineStage::FRAGMENT_SHADER) {
            result |= vk::AccessFlags::SHADER_WRITE;
        }

        if stage.contains(PipelineStage::EARLY_FRAGMENT_TESTS) {
            result |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }

        if stage.contains(PipelineStage::LATE_FRAGMENT_TESTS) {
            result |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }

        if stage.contains(PipelineStage::COLOR_ATTACHMENT_OUTPUT) {
            result |= vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        }

        if stage.contains(PipelineStage::COMPUTE_SHADER) {
            result |= vk::AccessFlags::SHADER_WRITE;
        }

        if stage.contains(PipelineStage::TRANSFER) {
            result |= vk::AccessFlags::TRANSFER_WRITE;
        }

        if stage.contains(PipelineStage::BOTTOM_OF_PIPE) {
            result |= vk::AccessFlags::MEMORY_WRITE;
        }

        if stage.contains(PipelineStage::HOST) {
            result |= vk::AccessFlags::HOST_WRITE;
        }

        if stage.contains(PipelineStage::ALL_GRAPHICS) {
            result |= vk::AccessFlags::MEMORY_WRITE;
        }

        if stage.contains(PipelineStage::ALL_COMMANDS) {
            result |= vk::AccessFlags::MEMORY_WRITE;
        }
    }

    result
}

pub enum Barrier {
    Image {
        image: usize,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
        src_access: Access,
        dst_access: Access,
        image_aspect: ImageAspect,
    },
    Buffer {
        buffer: usize,
        offset: usize,
        size: usize,
        src_access: Access,
        dst_access: Access,
    },
}

pub struct PipelineBarrier {
    pub src_stage: PipelineStage,
    pub dst_stage: PipelineStage,
    pub barriers: Vec<Barrier>,
}

pub struct PushConstant<'a, T> {
    pub data: T,
    pub pipeline: &'a Pipeline,
}

pub struct BindIndexBuffer {
    pub buffer: usize,
    pub offset: usize,
}

pub struct BufferWrite<'a, T: Copy> {
    pub buffer: usize,
    pub offset: usize,
    pub src: &'a [T],
}

pub struct BufferRead {
    pub buffer: usize,
    pub offset: usize,
    pub size: usize,
}

#[derive(Clone, Copy)]
pub struct Region {
    pub src: usize,
    pub dst: usize,
    pub size: usize,
}

#[derive(Clone)]
pub struct BufferCopy {
    pub from: usize,
    pub to: usize,
    pub regions: Vec<Region>,
}

pub struct BlitImage {
    pub from: usize,
    pub to: usize,
    pub src: (usize, usize, usize),
    pub dst: (usize, usize, usize),
}

pub struct ImageCopy {
    pub from: usize,
    pub to: usize,
    pub src: usize,
    pub dst: (usize, usize, usize),
    pub size: (usize, usize, usize),
}

pub struct BufferImageCopy {
    pub from: usize,
    pub to: usize,
    pub src: (usize, usize, usize),
    pub dst: usize,
    pub size: (usize, usize, usize),
}

pub struct Draw {
    pub vertex_count: usize,
    pub instance_count: usize,
    pub first_vertex: usize,
    pub first_instance: usize,
}

pub struct DrawIndirect {
    pub buffer: usize,
    pub offset: usize,
    pub draw_count: usize,
    pub stride: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[derive(Clone)]
pub enum WriteBinding {
    Buffer {
        buffer: Buffer,
        range: usize,
        offset: usize,
    },
    Image(Image),
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

pub struct DrawIndexed {
    pub index_count: usize,
}

pub struct Render {
    pub color: Vec<Attachment>,
    pub depth: Option<Attachment>,
    pub use_stencil: bool,
    pub render_area: RenderArea,
}

pub enum Clear {
    Color(f32, f32, f32, f32),
    Depth(f32),
    DepthStencil(f32, u8),
}

impl Default for Clear {
    fn default() -> Self {
        Self::Color(0.0, 0.0, 0.0, 1.0)
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp {
    #[default]
    Load,
    Clear,
    DontCare,
}

pub struct RenderPassBeginInfo<'a> {
    pub framebuffer: &'a Framebuffer,
    pub render_pass: &'a RenderPass,
    pub clear: Vec<Clear>,
    pub render_area: RenderArea,
}

impl From<LoadOp> for vk::AttachmentLoadOp {
    fn from(op: LoadOp) -> Self {
        match op {
            LoadOp::Load => Self::LOAD,
            LoadOp::Clear => Self::CLEAR,
            LoadOp::DontCare => Self::DONT_CARE,
        }
    }
}

pub struct Attachment {
    pub image: usize,
    pub load_op: LoadOp,
    pub clear: Clear,
}

impl Default for Attachment {
    fn default() -> Self {
        Self {
            image: usize::MAX,
            load_op: Default::default(),
            clear: Default::default(),
        }
    }
}

#[derive(Default)]
pub struct RenderArea {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Commands<'_> {
    ///This tells the render graph to send the tasks to the GPU.
    ///Without it, nothing will happen.
    pub fn submit(&mut self, submit: Submit) -> Result<()> {
        *self.submit = Some(submit);
        Ok(())
    }

    ///This tells the GPU to show what we drew to the screen.
    pub fn present(&mut self, present: Present) -> Result<()> {
        *self.present = Some(present);
        Ok(())
    }

    ///This tells the GPU to dispatch the currently bound compute pipeline.
    pub fn dispatch(&mut self, x: usize, y: usize, z: usize) -> Result<()> {
        let Commands {
            device,
            command_buffer,
            ..
        } = self;

        let DeviceInner { logical_device, .. } = &*device;

        unsafe { logical_device.cmd_dispatch(**command_buffer, x as _, y as _, z as _) };

        Ok(())
    }

    pub fn start_render_pass(&mut self, render_pass_info: RenderPassBeginInfo<'_>) -> Result<()> {
        let mut cv = vec![];

        for clear in render_pass_info.clear {
            cv.push(match clear {
                Clear::Color(r, g, b, a) => vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [r, g, b, a],
                    },
                },
                Clear::Depth(d) => vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: d,
                        stencil: 0,
                    },
                },
                Clear::DepthStencil(d, s) => vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: d,
                        stencil: s as _,
                    },
                },
            });
        }
        let begin_info = vk::RenderPassBeginInfo {
            render_pass: render_pass_info.render_pass.render_pass,
            framebuffer: render_pass_info.framebuffer.framebuffer,
            render_area: vk::Rect2D {
                offset: vk::Offset2D {
                    x: render_pass_info.render_area.x,
                    y: render_pass_info.render_area.y,
                },
                extent: vk::Extent2D {
                    width: render_pass_info.render_area.width,
                    height: render_pass_info.render_area.height,
                },
            },
            clear_value_count: cv.len() as _,
            p_clear_values: cv.as_ptr(),
            ..Default::default()
        };

        let Commands {
            device,
            command_buffer,
            ..
        } = self;

        unsafe {
            device.logical_device.cmd_begin_render_pass(
                **command_buffer,
                &begin_info,
                vk::SubpassContents::INLINE,
            )
        };

        Ok(())
    }

    pub fn end_render_pass(&mut self, render_pass: &'_ RenderPass) {
        let Commands {
            device,
            command_buffer,
            ..
        } = self;

        unsafe { device.logical_device.cmd_end_render_pass(**command_buffer) };
    }

    ///This sends information to the pipeline; this information hitches a ride with the command as it is sent to the GPU.
    ///It should be no more than 128 bytes.
    pub fn push_constant<'a, T: Copy>(&mut self, push_constant: PushConstant<'a, T>) -> Result<()> {
        let Commands {
            device,
            command_buffer,
            ..
        } = self;

        let DeviceInner { logical_device, .. } = &*device;

        let PushConstant { data, pipeline } = push_constant;

        let PipelineInner {
            bind_point, modify, ..
        } = &*pipeline.inner;

        let PipelineModify { layout, .. } = modify.lock().unwrap().clone();

        use crate::pipeline::PipelineBindPoint;

        let shader_stage = match bind_point {
            PipelineBindPoint::Graphics => {
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
            }
            PipelineBindPoint::Compute => vk::ShaderStageFlags::COMPUTE,
        };

        unsafe {
            logical_device.cmd_push_constants(
                **command_buffer,
                layout,
                shader_stage,
                0,
                slice::from_raw_parts(&data as *const _ as *const u8, mem::size_of::<T>()),
            );
        }

        Ok(())
    }

    ///Stops everything in the queue and writes to a buffer. If you use this in the middle of a render graph,
    ///it can create "bubbles" in the pipeline; essentially, instances where the gpu is waiting for work to be done.
    pub fn write_buffer<T: Copy>(&mut self, write: BufferWrite<T>) -> Result<()> {
        let Commands {
            device,
            qualifiers,
            command_buffer,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*device;

        let BufferWrite {
            buffer,
            offset,
            src,
        } = write;

        let resources = resources.lock().unwrap();

        let Qualifier::Buffer(buffer_handle, _) = qualifiers.get(buffer).ok_or(Error::ResourceNotFound)? else {
            Err(Error::InvalidResource)?
        };

        let InternalBuffer { memory, .. } = resources
            .buffers
            .get(*buffer_handle)
            .ok_or(Error::ResourceNotFound)?;

        let InternalMemory { memory, .. } = memory;

        let size = mem::size_of_val(src);

        let dst = unsafe {
            logical_device.map_memory(*memory, offset as _, size as _, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| Error::MemoryMapFailed)?;

        unsafe { slice::from_raw_parts_mut(dst as *mut T, src.len()) }.copy_from_slice(src);

        unsafe {
            logical_device.unmap_memory(*memory);
        }

        Ok(())
    }

    ///Stops everything in the queue and reads from a buffer. If you use this in the middle of a render graph,
    ///it can create "bubbles" in the pipeline; essentially, instances where the gpu is waiting for work to be done.
    pub fn read_buffer(&mut self, read: BufferRead) -> Result<Vec<u8>> {
        let Commands {
            device,
            qualifiers,
            command_buffer,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*device;

        let BufferRead {
            buffer,
            offset,
            size,
        } = read;

        let resources = resources.lock().unwrap();

        let Qualifier::Buffer(buffer_handle, _) = qualifiers.get(buffer).ok_or(Error::ResourceNotFound)? else {
            Err(Error::InvalidResource)?
        };

        let InternalBuffer { memory, .. } = resources
            .buffers
            .get(*buffer_handle)
            .ok_or(Error::ResourceNotFound)?;

        let InternalMemory { memory, .. } = memory;

        let src = unsafe {
            logical_device.map_memory(*memory, offset as _, size as _, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| {
            dbg!(e);
            Error::MemoryMapFailed
        })?;

        let mut dst = Vec::with_capacity(size);

        unsafe { ptr::copy(src as *const u8, dst.as_mut_ptr(), size) };
        unsafe { dst.set_len(size) };
        unsafe {
            logical_device.unmap_memory(*memory);
        }

        Ok(dst)
    }

    ///Set the resolution to render at.
    pub fn set_resolution(&mut self, resolution: (u32, u32), flip_y: bool) -> Result<()> {
        let (width, height) = resolution;

        let Commands {
            device,
            command_buffer,
            ..
        } = self;

        let DeviceInner { logical_device, .. } = &*device;

        let viewport = vk::Viewport {
            x: 0.0,
            y: if flip_y { height as f32 } else { 0.0 },
            width: width as f32,
            height: if flip_y {
                -(height as f32)
            } else {
                height as f32
            },
            min_depth: 0.0,
            max_depth: 1.0,
        };

        unsafe {
            logical_device.cmd_set_viewport(**command_buffer, 0, &[viewport]);
        }

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: width as _,
                height: height as _,
            },
        };

        unsafe {
            logical_device.cmd_set_scissor(**command_buffer, 0, &[scissor]);
        }

        Ok(())
    }

    ///Copy from one buffer to another.
    pub fn copy_buffer_to_buffer(&mut self, copy: BufferCopy) -> Result<()> {
        let Commands {
            device,
            qualifiers,
            command_buffer,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*device;

        let BufferCopy { from, to, regions } = copy;

        let resources = resources.lock().unwrap();

        let Qualifier::Buffer(from_buffer_handle, _) = qualifiers.get(from).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let InternalBuffer {
            buffer: from_buffer,
            ..
        } = resources
            .buffers
            .get(*from_buffer_handle)
            .ok_or(Error::ResourceNotFound)?;

        let Qualifier::Buffer(to_buffer_handle, _) = qualifiers.get(to).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let InternalBuffer {
            buffer: to_buffer, ..
        } = resources
            .buffers
            .get(*to_buffer_handle)
            .ok_or(Error::ResourceNotFound)?;

        let regions = regions
            .into_iter()
            .map(|x| vk::BufferCopy {
                src_offset: x.src as _,
                dst_offset: x.dst as _,
                size: x.size as _,
            })
            .collect::<Vec<_>>();

        unsafe {
            logical_device.cmd_copy_buffer(**command_buffer, *from_buffer, *to_buffer, &regions);
        }

        Ok(())
    }

    ///Copy from one buffer to an image.
    pub fn copy_buffer_to_image(&mut self, copy: ImageCopy) -> Result<()> {
        let Commands {
            device,
            qualifiers,
            command_buffer,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*device;

        let ImageCopy {
            from,
            to,
            src,
            dst,
            size,
        } = copy;

        let resources = resources.lock().unwrap();

        let Qualifier::Buffer(from_buffer_handle, _) = qualifiers.get(from).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let InternalBuffer {
            buffer: from_buffer,
            ..
        } = resources
            .buffers
            .get(*from_buffer_handle)
            .ok_or(Error::ResourceNotFound)?;

        let Qualifier::Image(to_image_handle, to_image_access, image_aspect) = qualifiers.get(to).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let to_image = resources
            .images
            .get(*to_image_handle)
            .ok_or(Error::ResourceNotFound)?
            .get_image();

        let to_image_format = resources
            .images
            .get(*to_image_handle)
            .ok_or(Error::ResourceNotFound)?
            .get_format();

        let regions = [vk::BufferImageCopy {
            buffer_offset: src as _,
            image_offset: vk::Offset3D {
                x: dst.0 as _,
                y: dst.1 as _,
                z: dst.2 as _,
            },
            image_extent: vk::Extent3D {
                width: size.0 as _,
                height: size.1 as _,
                depth: size.2 as _,
            },
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: (*image_aspect).into(),
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        }];

        unsafe {
            logical_device.cmd_copy_buffer_to_image(
                **command_buffer,
                *from_buffer,
                to_image,
                ImageLayout::from(*to_image_access).into(),
                &regions,
            );
        }

        Ok(())
    }

    ///Copy from an image to a buffer.
    pub fn copy_image_to_buffer(&mut self, copy: BufferImageCopy) -> Result<()> {
        let Commands {
            device,
            qualifiers,
            command_buffer,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*device;

        let BufferImageCopy {
            from,
            to,
            src,
            dst,
            size,
        } = copy;

        let resources = resources.lock().unwrap();

        let Qualifier::Buffer(from_buffer_handle, _) = qualifiers.get(to).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let InternalBuffer {
            buffer: from_buffer,
            ..
        } = resources
            .buffers
            .get(*from_buffer_handle)
            .ok_or(Error::ResourceNotFound)?;

        let Qualifier::Image(to_image_handle, to_image_access, image_aspect) = qualifiers.get(from).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let to_image = resources
            .images
            .get(*to_image_handle)
            .ok_or(Error::ResourceNotFound)?
            .get_image();

        let to_image_format = resources
            .images
            .get(*to_image_handle)
            .ok_or(Error::ResourceNotFound)?
            .get_format();

        let regions = [vk::BufferImageCopy {
            buffer_offset: dst as _,
            image_offset: vk::Offset3D {
                x: src.0 as _,
                y: src.1 as _,
                z: src.2 as _,
            },
            image_extent: vk::Extent3D {
                width: size.0 as _,
                height: size.1 as _,
                depth: size.2 as _,
            },
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: (*image_aspect).into(),
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        }];

        unsafe {
            logical_device.cmd_copy_image_to_buffer(
                **command_buffer,
                to_image,
                ImageLayout::from(*to_image_access).into(),
                *from_buffer,
                &regions,
            );
        }

        Ok(())
    }

    ///Copy from an image to a buffer.
    /// TODO make this more flexible and closer to vk api. I am being lazy rn.
    pub fn blit_image(&mut self, blit: BlitImage) -> Result<()> {
        let Commands {
            device,
            qualifiers,
            command_buffer,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*device;

        let BlitImage {
            from,
            to,
            src,
            dst
        } = blit;

        let resources = resources.lock().unwrap();

        let Qualifier::Image(from_image_handle, from_image_access, from_image_aspect) = qualifiers.get(from).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let from_image = resources
            .images
            .get(*from_image_handle)
            .ok_or(Error::ResourceNotFound)?
            .get_image();

        let from_image_format = resources
            .images
            .get(*from_image_handle)
            .ok_or(Error::ResourceNotFound)?
            .get_format();

        let Qualifier::Image(to_image_handle, to_image_access, image_aspect) = qualifiers.get(to).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let to_image = resources
            .images
            .get(*to_image_handle)
            .ok_or(Error::ResourceNotFound)?
            .get_image();

        let to_image_format = resources
            .images
            .get(*to_image_handle)
            .ok_or(Error::ResourceNotFound)?
            .get_format();

        let regions = [vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: (*from_image_aspect).into(),
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: (*from_image_aspect).into(),
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offsets: [Offset3D::default(), Offset3D { x: src.0 as _, y: src.1 as _, z: src.2 as _}],
            dst_offsets: [Offset3D::default(), Offset3D { x: dst.0 as _, y: dst.1 as _, z: dst.2 as _}],
        }];

        unsafe {
            logical_device.cmd_blit_image(
                **command_buffer,
                from_image,
                ImageLayout::from(*from_image_access).into(),
                to_image,
                ImageLayout::from(*to_image_access).into(),
                &regions,
                vk::Filter::NEAREST   
            )
        };

        Ok(())
    }

    ///Tell the GPU we would like to start rendering.
    ///After this command, set your pipeline and push constant, then draw.
    pub fn start_rendering(&mut self, render: Render) -> Result<()> {
        let Commands {
            device,
            qualifiers,
            command_buffer,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*device;

        let Render {
            color,
            depth,
            use_stencil,
            render_area,
        } = render;

        let resources = resources.lock().unwrap();

        let mut color_rendering_attachment_infos = vec![Default::default(); color.len()];

        for (i, color) in color.iter().enumerate() {
            let Qualifier::Image(color_handle, _, _) = qualifiers.get(color.image).ok_or(Error::InvalidResource)? else {
                Err(Error::InvalidResource)?
            };

            let image_view = resources
                .images
                .get(*color_handle)
                .ok_or(Error::ResourceNotFound)?
                .get_image_view();

            let Clear::Color(clear_r, clear_g, clear_b, clear_a) = color.clear else {
                Err(Error::InvalidAttachment)?
            };

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [clear_r, clear_g, clear_b, clear_a],
                },
            };

            color_rendering_attachment_infos[i] = vk::RenderingAttachmentInfoKHR {
                image_view,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                load_op: color.load_op.into(),
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value,
                ..Default::default()
            };
        }

        let depth_rendering_attachment_info = if let Some(depth) = depth {
            let Qualifier::Image(depth_handle, _, _) = qualifiers.get(depth.image).ok_or(Error::InvalidResource)? else {
                Err(Error::InvalidResource)?
            };

            let image_view = resources
                .images
                .get(*depth_handle)
                .ok_or(Error::ResourceNotFound)?
                .get_image_view();

            let clear_value = match depth.clear {
                Clear::Depth(clear_d) => vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: clear_d,
                        stencil: 0,
                    },
                },
                Clear::DepthStencil(clear_d, clear_s) => vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: clear_d,
                        stencil: clear_s as u32,
                    },
                },
                _ => Err(Error::InvalidAttachment)?,
            };

            Some(vk::RenderingAttachmentInfoKHR {
                image_view,
                image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                load_op: depth.load_op.into(),
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value,
                ..Default::default()
            })
        } else {
            None
        };

        let RenderArea {
            x,
            y,
            width,
            height,
        } = render_area;

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D { width, height },
        };

        let rendering_info = {
            let layer_count = 1;

            let color_attachment_count = color_rendering_attachment_infos.len() as _;

            let p_color_attachments = color_rendering_attachment_infos.as_ptr();

            let p_depth_attachment = depth_rendering_attachment_info
                .as_ref()
                .map(|r| r as *const _)
                .unwrap_or(ptr::null());

            let p_stencil_attachment = if use_stencil {
                p_depth_attachment
            } else {
                ptr::null()
            };

            vk::RenderingInfoKHR {
                render_area,
                layer_count,
                color_attachment_count,
                p_color_attachments,
                p_depth_attachment,
                p_stencil_attachment,
                ..Default::default()
            }
        };

        unsafe {
            logical_device.cmd_begin_rendering(**command_buffer, &rendering_info);
        }

        Ok(())
    }

    ///Tell the GPU we would like to stop rendering.
    pub fn end_rendering(&mut self) -> Result<()> {
        let Commands {
            device,
            command_buffer,
            ..
        } = self;

        let DeviceInner { logical_device, .. } = &*device;

        unsafe {
            logical_device.cmd_end_rendering(**command_buffer);
        }

        Ok(())
    }

    pub fn write_bindings(
        &mut self,
        pipeline: &Pipeline,
        bindings: Vec<WriteBinding>,
    ) -> Result<()> {
        let Commands {
            device,
            command_buffer,
            swapchain,
            ..
        } = self;

        let resources = device.resources.lock().unwrap();

        let internal_swapchain = resources
            .swapchains
            .get(**swapchain)
            .ok_or(Error::InvalidResource)?;

        let mut buffer_infos = HashMap::<usize, vk::DescriptorBufferInfo>::new();
        let mut image_infos = HashMap::<usize, vk::DescriptorImageInfo>::new();

        for (i, binding) in bindings.iter().enumerate() {
            match binding {
                WriteBinding::Buffer {
                    buffer,
                    range,
                    offset,
                } => {
                    let internal_buffer = resources
                        .buffers
                        .get(*buffer)
                        .ok_or(Error::InvalidResource)?;

                    buffer_infos.insert(
                        i,
                        vk::DescriptorBufferInfo {
                            buffer: internal_buffer.buffer,
                            offset: *offset as u64,
                            range: *range as u64,
                        },
                    );
                }
                WriteBinding::Image(image) => {
                    let internal_image =
                        resources.images.get(*image).ok_or(Error::InvalidResource)?;
                    image_infos.insert(
                        i,
                        vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: internal_image.get_image_view(),
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                    );
                }
            }
        }

        let descriptor_writes = bindings
            .into_iter()
            .enumerate()
            .map(|(i, binding)| {
                let buffer = buffer_infos
                    .get(&i)
                    .map(|r| r as *const _)
                    .unwrap_or(ptr::null());
                let image = image_infos
                    .get(&i)
                    .map(|r| r as *const _)
                    .unwrap_or(ptr::null());

                vk::WriteDescriptorSet {
                    dst_set: pipeline.inner.modify.lock().unwrap().descriptor_sets
                        [internal_swapchain.current_frame],
                    dst_binding: i as _,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: match binding {
                        WriteBinding::Buffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
                        WriteBinding::Image(_) => vk::DescriptorType::STORAGE_IMAGE,
                    },
                    p_buffer_info: buffer,
                    p_image_info: image,
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>();

        unsafe {
            device
                .logical_device
                .update_descriptor_sets(&descriptor_writes, &[])
        };
        Ok(())
    }

    pub fn set_pipeline(&mut self, pipeline: &Pipeline) -> Result<()> {
        let Commands {
            device,
            command_buffer,
            swapchain,
            ..
        } = self;

        let DeviceInner { logical_device, .. } = &*device;

        let PipelineInner {
            bind_point, modify, ..
        } = &*pipeline.inner;

        let PipelineModify {
            layout,
            pipeline,
            descriptor_sets,
            ..
        } = modify.lock().unwrap().clone();

        let bind_point = vk::PipelineBindPoint::from(*bind_point);

        unsafe {
            logical_device.cmd_bind_pipeline(**command_buffer, bind_point, pipeline);
        }

        let mut resources = device.resources.lock().unwrap();

        let internal_swapchain = resources
            .swapchains
            .get(**swapchain)
            .ok_or(Error::InvalidResource)?;

        unsafe {
            logical_device.cmd_bind_descriptor_sets(
                **command_buffer,
                bind_point,
                layout,
                0,
                &[descriptor_sets[internal_swapchain.last_acquisition_index.unwrap() as usize]],
                &[],
            );
        }

        Ok(())
    }

    pub fn draw_indirect(&mut self, draw_indirect: DrawIndirect) -> Result<()> {
        let Commands {
            device,
            command_buffer,
            qualifiers,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*device;

        let DrawIndirect {
            buffer,
            offset,
            draw_count,
            stride,
        } = draw_indirect;

        let Qualifier::Buffer(buffer_handle, _) = qualifiers.get(buffer).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let buffer = resources
            .lock()
            .unwrap()
            .buffers
            .get(*buffer_handle)
            .ok_or(Error::ResourceNotFound)?
            .buffer;

        unsafe {
            logical_device.cmd_draw_indirect(
                **command_buffer,
                buffer,
                offset as u64,
                draw_count as u32,
                stride as u32,
            );
        }

        Ok(())
    }

    pub fn draw_indexed_indirect(&mut self, draw_indirect: DrawIndirect) -> Result<()> {
        let Commands {
            device,
            command_buffer,
            qualifiers,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*device;

        let DrawIndirect {
            buffer,
            offset,
            draw_count,
            stride,
        } = draw_indirect;

        let Qualifier::Buffer(buffer_handle, _) = qualifiers.get(buffer).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let buffer = resources
            .lock()
            .unwrap()
            .buffers
            .get(*buffer_handle)
            .ok_or(Error::ResourceNotFound)?
            .buffer;

        unsafe {
            logical_device.cmd_draw_indexed_indirect(
                **command_buffer,
                buffer,
                offset as u64,
                draw_count as u32,
                stride as u32,
            );
        }

        Ok(())
    }

    ///Draw using vertex count
    pub fn draw(&mut self, draw: Draw) -> Result<()> {
        let Commands {
            device,
            command_buffer,
            ..
        } = self;

        let DeviceInner { logical_device, .. } = &*device;

        let Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        } = draw;

        unsafe {
            logical_device.cmd_draw(
                **command_buffer,
                vertex_count as _,
                instance_count as _,
                first_vertex as _,
                first_instance as _,
            );
        }

        Ok(())
    }

    ///Sets-up synchronization of commands in the same task.
    pub fn pipeline_barrier(&mut self, pipeline_barrier: PipelineBarrier) -> Result<()> {
        let PipelineBarrier {
            src_stage,
            dst_stage,
            barriers,
        } = pipeline_barrier;

        let Commands {
            device,
            qualifiers,
            command_buffer,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*device;

        let resources = resources.lock().unwrap();

        let memory_barriers = vec![];

        let mut buffer_barriers = vec![];

        let mut image_barriers = vec![];

        for barrier in barriers {
            match barrier {
                Barrier::Image {
                    image,
                    old_layout,
                    new_layout,
                    src_access,
                    dst_access,
                    image_aspect,
                } => {
                    let Qualifier::Image(image_handle, image_access, image_aspect) = qualifiers.get(image).ok_or(Error::InvalidResource)? else {
                    Err(Error::InvalidResource)?
                };

                    let image = resources
                        .images
                        .get(*image_handle)
                        .ok_or(Error::ResourceNotFound)?
                        .get_image();

                    let format = resources
                        .images
                        .get(*image_handle)
                        .ok_or(Error::ResourceNotFound)?
                        .get_format();

                    let src_access_mask = stage_access_to_access((src_stage, src_access));

                    let dst_access_mask = stage_access_to_access((dst_stage, dst_access));

                    let old_layout = old_layout.into();

                    let new_layout = new_layout.into();

                    let subresource_range = vk::ImageSubresourceRange {
                        aspect_mask: (*image_aspect).into(),
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };

                    image_barriers.push(vk::ImageMemoryBarrier {
                        image,
                        src_access_mask,
                        dst_access_mask,
                        old_layout,
                        new_layout,
                        subresource_range,
                        ..Default::default()
                    });
                }
                Barrier::Buffer {
                    buffer,
                    size,
                    offset,
                    src_access,
                    dst_access,
                } => {
                    let Qualifier::Buffer(buffer_handle, _) = qualifiers.get(buffer).ok_or(Error::InvalidResource)? else {
                    Err(Error::InvalidResource)?
                };

                    let buffer = resources
                        .buffers
                        .get(*buffer_handle)
                        .ok_or(Error::ResourceNotFound)?
                        .buffer;

                    let size = size as _;

                    let offset = offset as _;

                    let src_access_mask = stage_access_to_access((src_stage, src_access));

                    let dst_access_mask = stage_access_to_access((dst_stage, dst_access));

                    buffer_barriers.push(vk::BufferMemoryBarrier {
                        buffer,
                        size,
                        offset,
                        src_access_mask,
                        dst_access_mask,
                        ..Default::default()
                    });
                }
            }
        }

        unsafe {
            logical_device.cmd_pipeline_barrier(
                **command_buffer,
                src_stage.into(),
                dst_stage.into(),
                Default::default(),
                &memory_barriers,
                &buffer_barriers,
                &image_barriers,
            );
        }

        Ok(())
    }
}
