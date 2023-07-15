use crate::prelude::*;

use ash::vk;

use bitflags::bitflags;

pub(crate) enum InternalImage {
    Managed {
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        format: Format,
    },
    Swapchain {
        image: vk::Image,
        view: vk::ImageView,
        format: Format,
    },
}
impl InternalImage {
    pub(crate) fn get_image(&self) -> vk::Image {
        match self {
            Self::Managed { image, .. } => *image,
            Self::Swapchain { image, .. } => *image,
        }
    }
    pub(crate) fn get_image_view(&self) -> vk::ImageView {
        match self {
            Self::Managed { view, .. } => *view,
            Self::Swapchain { view, .. } => *view,
        }
    }
    pub(crate) fn get_format(&self) -> Format {
        match self {
            Self::Managed { format, .. } => *format,
            Self::Swapchain { format, .. } => *format,
        }
    }
}

#[derive(Clone, Copy)]
pub enum ImageExtent {
    OneDim(usize),
    TwoDim(usize, usize),
    ThreeDim(usize, usize, usize),
}

impl Default for ImageExtent {
    fn default() -> Self {
        Self::TwoDim(1, 1)
    }
}

pub struct ImageInfo<'a> {
    pub extent: ImageExtent,
    pub usage: ImageUsage,
    pub format: Format,
    pub debug_name: &'a str,
}

impl Default for ImageInfo<'_> {
    fn default() -> Self {
        Self {
            extent: Default::default(),
            usage: ImageUsage::empty(),
            format: Format::Undefined,
            debug_name: "",
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[repr(transparent)]
///An image handle, used for bindless images.
pub struct Image(pub u32);

impl From<Image> for u32 {
    fn from(handle: Image) -> Self {
        handle.0
    }
}

impl From<u32> for Image {
    fn from(handle: u32) -> Self {
        Self(handle)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ImageLayout {
    Undefined,
    General,
    ReadOnlyOptimal,
    TransferSrcOptimal,
    TransferDstOptimal,
    AttachmentOptimal,
    ColorAttachmentOptimal,
    DepthStencilAttachmentOptimal,
    Present,
}

impl From<ImageLayout> for vk::ImageLayout {
    fn from(layout: ImageLayout) -> Self {
        match layout {
            ImageLayout::Undefined => Self::UNDEFINED,
            ImageLayout::ReadOnlyOptimal => Self::READ_ONLY_OPTIMAL,
            ImageLayout::TransferSrcOptimal => Self::TRANSFER_SRC_OPTIMAL,
            ImageLayout::TransferDstOptimal => Self::TRANSFER_DST_OPTIMAL,
            ImageLayout::General => Self::GENERAL,
            ImageLayout::AttachmentOptimal => Self::ATTACHMENT_OPTIMAL,
            ImageLayout::ColorAttachmentOptimal => Self::COLOR_ATTACHMENT_OPTIMAL,
            ImageLayout::DepthStencilAttachmentOptimal => Self::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ImageLayout::Present => Self::PRESENT_SRC_KHR,
        }
    }
}

bitflags! {
    #[derive(Clone, Copy)]
    pub struct ImageUsage: u32 {
        const TRANSFER_SRC = 0x00000001;
        const TRANSFER_DST = 0x00000002;
        const COLOR = 0x00000010;
        const DEPTH_STENCIL = 0x00000020;
    }
}

bitflags! {
    #[derive(Clone, Copy)]
    pub struct ImageAspect: u32 {
        const COLOR = 0x00000001;
        const DEPTH = 0x00000002;
        const STENCIL = 0x00000004;
    }
}

impl From<ImageAspect> for vk::ImageAspectFlags {
    fn from(aspect: ImageAspect) -> Self {
        let mut result = vk::ImageAspectFlags::empty();

        if aspect.contains(ImageAspect::COLOR) {
            result |= vk::ImageAspectFlags::COLOR;
        }

        if aspect.contains(ImageAspect::DEPTH) {
            result |= vk::ImageAspectFlags::DEPTH;
        }

        if aspect.contains(ImageAspect::STENCIL) {
            result |= vk::ImageAspectFlags::STENCIL;
        }

        result
    }
}

impl Default for ImageAspect {
    fn default() -> Self {
        Self::COLOR
    }
}

impl From<ImageUsage> for vk::ImageUsageFlags {
    fn from(usage: ImageUsage) -> Self {
        let mut result = vk::ImageUsageFlags::empty();

        if usage.contains(ImageUsage::TRANSFER_SRC) {
            result |= vk::ImageUsageFlags::TRANSFER_SRC;
        }

        if usage.contains(ImageUsage::TRANSFER_DST) {
            result |= vk::ImageUsageFlags::TRANSFER_DST;
        }

        if usage.contains(ImageUsage::COLOR) {
            result |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }

        if usage.contains(ImageUsage::DEPTH_STENCIL) {
            result |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }

        result
    }
}
