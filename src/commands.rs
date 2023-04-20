use crate::device::DeviceInner;
use crate::pipeline::{PipelineInner, PipelineModify};
use crate::prelude::*;

use std::default::default;
use std::mem;
use std::ops;
use std::ptr;
use std::slice;
use std::sync::Mutex;

use ash::vk;

use bitflags::bitflags;

pub struct Commands<'a> {
    pub(crate) device: &'a DeviceInner,
    pub(crate) qualifiers: &'a [Qualifier],
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

pub struct PushConstant<'a, T: Copy, const S: usize, const C: usize> {
    pub data: T,
    pub pipeline: &'a Pipeline<'a, S, C>,
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
}

pub struct BufferCopy {
    pub from: usize,
    pub to: usize,
    pub src: usize,
    pub dst: usize,
    pub size: usize,
}

pub struct ImageCopy {
    pub from: usize,
    pub to: usize,
    pub src: usize,
    pub dst: (usize, usize, usize),
    pub size: (usize, usize, usize),
}

pub struct Draw {
    pub vertex_count: usize,
}

pub struct DrawIndexed {
    pub index_count: usize,
}

pub struct Render<const N: usize> {
    pub color: [Attachment; N],
    pub depth: Option<Attachment>,
    pub render_area: RenderArea,
}

pub enum Clear {
    Color(f32, f32, f32, f32),
    Depth(f32),
}

impl Default for Clear {
    fn default() -> Self {
        Self::Color(0.0, 0.0, 0.0, 1.0)
    }
}

#[derive(Default, Clone, Copy)]
pub enum LoadOp {
    #[default]
    Load,
    Clear,
    DontCare,
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
            load_op: default(),
            clear: default(),
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
    pub fn submit(&mut self, submit: Submit) -> Result<()> {
        *self.submit = Some(submit);
        Ok(())
    }

    pub fn present(&mut self, present: Present) -> Result<()> {
        *self.present = Some(present);
        Ok(())
    }

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

    pub fn push_constant<'a, T: Copy, const S: usize, const C: usize>(
        &mut self,
        push_constant: PushConstant<'a, T, S, C>,
    ) -> Result<()> {
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
        .map_err(|_| Error::MemoryMapFailed)?;

        unsafe { slice::from_raw_parts_mut(dst as *mut T, src.len()) }.copy_from_slice(src);

        unsafe {
            logical_device.unmap_memory(*memory);
        }

        Ok(())
    }

    pub fn read_buffer<T: Copy>(&mut self, read: BufferRead) -> Result<T> {
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

        let BufferRead { buffer, offset } = read;

        let resources = resources.lock().unwrap();

        let Qualifier::Buffer(buffer_handle, _) = qualifiers.get(buffer).ok_or(Error::ResourceNotFound)? else {
            Err(Error::InvalidResource)?
        };

        let InternalBuffer { memory, .. } = resources
            .buffers
            .get(*buffer_handle)
            .ok_or(Error::ResourceNotFound)?;

        let InternalMemory { memory, .. } = memory;

        let size = mem::size_of::<T>();

        let src = unsafe {
            logical_device.map_memory(*memory, offset as _, size as _, vk::MemoryMapFlags::empty())
        }
        .map_err(|_| Error::MemoryMapFailed)?;

        let dst = unsafe { ptr::read::<T>(src as *const T) };

        unsafe {
            logical_device.unmap_memory(*memory);
        }

        Ok(dst)
    }

    pub fn set_resolution(&mut self, resolution: (u32, u32)) -> Result<()> {
        let (width, height) = resolution;

        let Commands {
            device,
            command_buffer,
            ..
        } = self;

        let DeviceInner { logical_device, .. } = &*device;

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
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

        let BufferCopy {
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

        let Qualifier::Buffer(to_buffer_handle, _) = qualifiers.get(to).ok_or(Error::InvalidResource)? else {
            Err(Error::InvalidResource)?
        };

        let InternalBuffer {
            buffer: to_buffer, ..
        } = resources
            .buffers
            .get(*to_buffer_handle)
            .ok_or(Error::ResourceNotFound)?;

        let regions = [vk::BufferCopy {
            src_offset: src as _,
            dst_offset: dst as _,
            size: size as _,
        }];

        unsafe {
            logical_device.cmd_copy_buffer(**command_buffer, *from_buffer, *to_buffer, &regions);
        }

        Ok(())
    }

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

        let Qualifier::Image(to_image_handle, to_image_access) = qualifiers.get(to).ok_or(Error::InvalidResource)? else {
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
                aspect_mask: to_image_format.into(),
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..default()
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

    pub fn start_rendering<const N: usize>(&mut self, render: Render<N>) -> Result<()> {
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
            render_area,
        } = render;

        let resources = resources.lock().unwrap();

        let mut color_rendering_attachment_infos = [default(); N];

        for (i, color) in color.iter().enumerate() {
            let Qualifier::Image(color_handle, _) = qualifiers.get(color.image).ok_or(Error::InvalidResource)? else {
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
                ..default()
            };
        }

        let depth_rendering_attachment_info = if let Some(depth) = depth {
            let Qualifier::Image(depth_handle, _) = qualifiers.get(depth.image).ok_or(Error::InvalidResource)? else {
                Err(Error::InvalidResource)?
            };

            let image_view = resources
                .images
                .get(*depth_handle)
                .ok_or(Error::ResourceNotFound)?
                .get_image_view();

            let Clear::Depth(clear_d) = depth.clear else {
                Err(Error::InvalidAttachment)?
            };

            let clear_value = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: clear_d,
                    stencil: 0,
                },
            };

            Some(vk::RenderingAttachmentInfoKHR {
                image_view,
                image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                load_op: depth.load_op.into(),
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value,
                ..default()
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

            vk::RenderingInfoKHR {
                render_area,
                layer_count,
                color_attachment_count,
                p_color_attachments,
                p_depth_attachment,
                ..default()
            }
        };

        unsafe {
            logical_device.cmd_begin_rendering(**command_buffer, &rendering_info);
        }

        Ok(())
    }

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

    pub fn set_pipeline<'a, const S: usize, const C: usize>(
        &mut self,
        pipeline: &'a Pipeline<'a, S, C>,
    ) -> Result<()> {
        let Commands {
            device,
            command_buffer,
            ..
        } = self;

        let DeviceInner {
            logical_device,
            descriptor_set,
            ..
        } = &*device;

        let PipelineInner {
            bind_point, modify, ..
        } = &*pipeline.inner;

        let PipelineModify { layout, pipeline } = modify.lock().unwrap().clone();

        let bind_point = vk::PipelineBindPoint::from(*bind_point);

        unsafe {
            logical_device.cmd_bind_pipeline(**command_buffer, bind_point, pipeline);
        }

        unsafe {
            logical_device.cmd_bind_descriptor_sets(
                **command_buffer,
                bind_point,
                layout,
                0,
                &[*descriptor_set],
                &[],
            );
        }

        Ok(())
    }

    pub fn draw(&mut self, draw: Draw) -> Result<()> {
        let Commands {
            device,
            command_buffer,
            ..
        } = self;

        let DeviceInner { logical_device, .. } = &*device;

        let Draw { vertex_count } = draw;

        unsafe {
            logical_device.cmd_draw(**command_buffer, vertex_count as _, 1, 0, 0);
        }

        Ok(())
    }

    pub fn pipeline_barrier(
        &mut self,
        pipeline_barrier: PipelineBarrier,
    ) -> Result<()> {
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
                } => {
                    let Qualifier::Image(image_handle, image_access) = qualifiers.get(image).ok_or(Error::InvalidResource)? else {
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
                        aspect_mask: format.into(),
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
                        ..default()
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
                        ..default()
                    });
                }
            }
        }

        unsafe {
            logical_device.cmd_pipeline_barrier(
                **command_buffer,
                src_stage.into(),
                dst_stage.into(),
                default(),
                &memory_barriers,
                &buffer_barriers,
                &image_barriers,
            );
        }

        Ok(())
    }
}
