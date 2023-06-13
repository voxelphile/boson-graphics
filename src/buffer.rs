pub use crate::prelude::*;

use ash::vk;

use bitflags::bitflags;

bitflags! {
    #[derive(Clone, Copy,)]
    pub struct BufferUsage: u32 {
        const TRANSFER_SRC = 0x00000001;
        const TRANSFER_DST = 0x00000002;
        const STORAGE = 0x00000004;
        const INDIRECT = 0x00000008;
    }
}

impl From<BufferUsage> for vk::BufferUsageFlags {
    fn from(usage: BufferUsage) -> Self {
        let mut result = vk::BufferUsageFlags::empty();

        if usage.contains(BufferUsage::TRANSFER_SRC) {
            result |= vk::BufferUsageFlags::TRANSFER_SRC;
        }

        if usage.contains(BufferUsage::TRANSFER_DST) {
            result |= vk::BufferUsageFlags::TRANSFER_DST;
        }

        if usage.contains(BufferUsage::STORAGE) {
            result |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }

        if usage.contains(BufferUsage::INDIRECT) {
            result |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }

        result
    }
}

pub(crate) struct InternalBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) size: usize,
    pub(crate) usage: BufferUsage,
    pub(crate) memory: InternalMemory,
    pub(crate) debug_name: String,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
#[repr(transparent)]
///A buffer handle, used for bindless buffers.
pub struct Buffer(pub(crate) u32);

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct BufferAddress(pub(crate) u64);

impl From<Buffer> for u32 {
    fn from(handle: Buffer) -> Self {
        handle.0
    }
}

impl From<u32> for Buffer {
    fn from(handle: u32) -> Self {
        Self(handle)
    }
}

pub struct BufferInfo<'a> {
    pub size: usize,
    pub memory: Memory,
    pub usage: BufferUsage,
    pub debug_name: &'a str,
}

impl Default for BufferInfo<'_> {
    fn default() -> Self {
        Self {
            size: 0,
            memory: Memory::DEVICE_LOCAL,
            usage: BufferUsage::all(),
            debug_name: "Buffer",
        }
    }
}
