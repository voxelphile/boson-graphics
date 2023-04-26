use std::mem;

use ash::vk;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    #[default]
    Undefined,
    R8Uint,
    R8Unorm,
    R32Uint,
    R32Sfloat,
    Rgba32Sfloat,
    Rg32Uint,
    R16Uint,
    Rgba8Uint,
    Rgb32Uint,
    Rgba8Unorm,
    Rgba8Srgb,
    Rgba32Uint,
    Bgra8Unorm,
    Bgra8Srgb,
    D32Sfloat,
    D32SfloatS8Uint
}

impl Format {
    pub(crate) fn entire_aspect(&self) -> vk::ImageAspectFlags {
        match self {
            Format::D32Sfloat => vk::ImageAspectFlags::DEPTH,
            Format::D32SfloatS8Uint => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            _ => vk::ImageAspectFlags::COLOR,
        }
    }
    pub(crate) fn is_depth_or_stencil(&self) -> bool {
        match self {
            Format::D32Sfloat 
            | Format::D32SfloatS8Uint => true,
            _ => false,
        }
    }
}

impl TryFrom<vk::Format> for Format {
    type Error = ();

    fn try_from(format: vk::Format) -> Result<Self, Self::Error> {
        use Format::*;

        Ok(match format {
            vk::Format::UNDEFINED => Undefined,
            vk::Format::R8_UINT => R8Uint,
            vk::Format::R8_UNORM => R8Unorm,
            vk::Format::R32_UINT => R32Uint,
            vk::Format::R32G32_UINT => Rg32Uint,
            vk::Format::R16_UINT => R32Uint,
            vk::Format::R32_SFLOAT => R32Sfloat,
            vk::Format::R8G8B8A8_UINT => Rgba8Uint,
            vk::Format::R32G32B32A32_UINT => Rgba32Uint,
            vk::Format::R32G32B32_UINT => Rgb32Uint,
            vk::Format::R32G32B32A32_SFLOAT => Rgba32Sfloat,
            vk::Format::R8G8B8A8_UNORM => Rgba8Unorm,
            vk::Format::R8G8B8A8_SRGB => Rgba8Srgb,
            vk::Format::B8G8R8A8_UNORM => Bgra8Unorm,
            vk::Format::B8G8R8A8_SRGB => Bgra8Srgb,
            vk::Format::D32_SFLOAT => D32Sfloat,
            vk::Format::D32_SFLOAT_S8_UINT => D32SfloatS8Uint,
            _ => Err(())?,
        })
    }
}

impl From<Format> for vk::Format {
    fn from(format: Format) -> Self {
        use Format::*;

        match format {
            Undefined => Self::UNDEFINED,
            R8Uint => Self::R8_UINT,
            R8Unorm => Self::R8_UNORM,
            R32Uint => Self::R32_UINT,
            Rg32Uint => Self::R32G32_UINT,
            R32Sfloat => Self::R32_SFLOAT,
            R16Uint => Self::R16_UINT,
            Rgb32Uint => Self::R32G32B32_UINT,
            Rgba32Sfloat => Self::R32G32B32A32_SFLOAT,
            Rgba32Uint => Self::R32G32B32A32_UINT,
            Rgba8Uint => Self::R8G8B8A8_UINT,
            Rgba8Unorm => Self::R8G8B8A8_UNORM,
            Rgba8Srgb => Self::R8G8B8A8_SRGB,
            Bgra8Unorm => Self::B8G8R8A8_UNORM,
            Bgra8Srgb => Self::B8G8R8A8_SRGB,
            D32Sfloat => Self::D32_SFLOAT,
            D32SfloatS8Uint => Self::D32_SFLOAT_S8_UINT
        }
    }
}
