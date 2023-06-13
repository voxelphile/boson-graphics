pub mod buffer;
pub mod commands;
pub mod context;
pub mod device;
pub mod format;
pub mod image;
pub mod memory;
pub mod pipeline;
pub mod semaphore;
pub mod swapchain;
pub mod task;

use std::error;
use std::fmt;
use std::result;

pub mod prelude {
    pub(crate) use crate::buffer::InternalBuffer;
    pub use crate::buffer::{Buffer, BufferAddress, BufferInfo, BufferUsage};
    pub use crate::commands::{
        Access, Attachment, Barrier, BindIndexBuffer, BufferCopy, BufferImageCopy, BufferRead,
        BufferWrite, Clear, Commands, Draw, DrawIndexed, DrawIndirect, DrawIndirectCommand,
        ImageCopy, LoadOp, PipelineBarrier, PushConstant, Render, RenderArea,
    };
    pub(crate) use crate::context::DESCRIPTOR_COUNT;
    pub use crate::context::{Context, ContextInfo};
    pub(crate) use crate::device::DeviceResources;
    pub use crate::device::{Device, DeviceInfo, Features};
    pub use crate::format::Format;
    pub(crate) use crate::image::InternalImage;
    pub use crate::image::{Image, ImageAspect, ImageExtent, ImageInfo, ImageLayout, ImageUsage};
    pub(crate) use crate::memory::InternalMemory;
    pub use crate::memory::Memory;
    pub use crate::pipeline::{
        Blend, BlendFactor, BlendOp, Color, ColorComponent, CompareOp, ComputePipelineInfo, Define,
        Depth, FaceCull, FrontFace, GraphicsPipelineInfo, Pipeline, PipelineCompiler,
        PipelineCompilerInfo, PipelineStage, PolygonMode, Raster, Shader, ShaderCompiler,
        ShaderType, Stencil, StencilOp, StencilState,
    };
    pub use crate::semaphore::{
        BinarySemaphore, BinarySemaphoreInfo, TimelineSemaphore, TimelineSemaphoreInfo,
    };
    pub(crate) use crate::swapchain::InternalSwapchain;
    pub use crate::swapchain::{Acquire, PresentMode, Swapchain, SwapchainInfo};
    pub(crate) use crate::task::Qualifier;
    pub use crate::task::{
        BufferAccess, ImageAccess, Present, RenderGraph, RenderGraphBuilder, RenderGraphInfo,
        Resource, Submit, Task,
    };
    pub(crate) use crate::{Error, Result};
}

#[derive(Debug)]
pub enum Error {
    Creation,
    ShaderCompilerNotFound,
    ShaderCompilationError { message: String },
    ResourceNotFound,
    InvalidResource,
    MemoryMapFailed,
    InvalidAttachment,
    FailedToAcquire,
    CreateBuffer,
    BindBufferMemory,
    AllocateMemory,
    AllocateDescriptorSets,
    EnumeratePhysicalDevices,
    CreateSurface,
    CreateCommandPool,
    CreateLogicalDevice,
    CreateDescriptorPool,
    CreateDescriptorSetLayout,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for Error {}

pub type Result<T> = result::Result<T, Error>;
