use crate::prelude::*;

use std::ops;

use ash::extensions::{ext, khr};
use ash::{vk, Entry, Instance};

pub fn default_surface_format_selector(format: Format) -> usize {
    use Format::*;
    match format {
        Rgba8Srgb => 90,
        Rgba8Unorm => 80,
        Bgra8Srgb => 70,
        Bgra8Unorm => 60,
        _ => 0,
    }
}

pub trait SurfaceFormatSelector = ops::Fn(Format) -> usize;

#[derive(Debug, Clone, Copy)]
pub enum PresentMode {
    DoNotWaitForVBlank,
    TripleBufferWaitForVBlank,
    DoubleBufferWaitForVBlank,
    DoubleBufferWaitForVBlankRelaxed,
}

impl From<PresentMode> for vk::PresentModeKHR {
    fn from(present_mode: PresentMode) -> Self {
        use PresentMode::*;

        match present_mode {
            DoNotWaitForVBlank => Self::IMMEDIATE,
            TripleBufferWaitForVBlank | DoubleBufferWaitForVBlank => Self::FIFO,
            DoubleBufferWaitForVBlankRelaxed => Self::FIFO_RELAXED,
        }
    }
}

pub struct SwapchainInfo<'a> {
    pub present_mode: PresentMode,
    pub image_usage: ImageUsage,
    pub width: u32,
    pub height: u32,
    pub surface_format_selector: &'a dyn SurfaceFormatSelector,
    pub old_swapchain: Option<Swapchain>,
    pub debug_name: &'a str,
}

impl Default for SwapchainInfo<'_> {
    fn default() -> Self {
        Self {
            present_mode: PresentMode::DoNotWaitForVBlank,
            image_usage: ImageUsage::TRANSFER_DST,
            width: 960,
            height: 540,
            surface_format_selector: &default_surface_format_selector,
            old_swapchain: None,
            debug_name: "Swapchain",
        }
    }
}

pub struct InternalSwapchain {
    pub(crate) format: Format,
    pub(crate) loader: khr::Swapchain,
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) images: Vec<Image>,
    pub(crate) last_acquisition_index: Option<u32>,
    pub(crate) current_frame: usize,
    pub(crate) allow_acquisition: bool,
}

#[derive(Clone, Copy)]
pub struct Swapchain(pub(crate) u32);

impl From<Swapchain> for u32 {
    fn from(handle: Swapchain) -> Self {
        handle.0
    }
}

impl From<u32> for Swapchain {
    fn from(handle: u32) -> Self {
        Self(handle)
    }
}

pub struct Acquire {
    pub swapchain: Swapchain,
    pub semaphore: Option<BinarySemaphore>,
}
