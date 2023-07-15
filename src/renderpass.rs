use ash::vk::{self};

use crate::{
    buffer::LoadOp,
    commands::Attachment,
    format::Format,
    image::{Image, ImageLayout},
};

#[derive(Clone)]
pub struct RenderPassAttachment {
    pub image: usize,
    pub load_op: LoadOp,
    pub format: Format,
    pub initial_layout: ImageLayout,
    pub final_layout: ImageLayout,
}

pub struct RenderPassInfo {
    pub color: Vec<RenderPassAttachment>,
    pub depth: Option<RenderPassAttachment>,
    pub stencil_load_op: LoadOp,
}

#[derive(Clone)]
pub struct RenderPass {
    pub(crate) render_pass: vk::RenderPass,
}

pub struct FramebufferInfo {
    pub render_pass: RenderPass,
    pub attachments: Vec<Image>,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone)]
pub struct Framebuffer {
    pub(crate) framebuffer: vk::Framebuffer,
}
