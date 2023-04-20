use crate::prelude::*;

use ash::vk;

use bitflags::bitflags;

bitflags! {
    pub struct Memory: u32 {
        const DEVICE_LOCAL = 0x00000001;
        const HOST_ACCESS = 0x00000002;
    }
}

pub(crate) struct InternalMemory {
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) properties: Memory,
}

impl From<Memory> for vk::MemoryPropertyFlags {
    fn from(memory: Memory) -> Self {
        let mut result = vk::MemoryPropertyFlags::empty();

        if memory.contains(Memory::DEVICE_LOCAL) {
            result |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
        }

        if memory.contains(Memory::HOST_ACCESS) {
            result |=
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        }

        result
    }
}

pub(crate) fn type_index(
    requirements: &vk::MemoryRequirements,
    properties: &vk::PhysicalDeviceMemoryProperties,
    memory: vk::MemoryPropertyFlags,
) -> Result<u32> {
    properties.memory_types[..properties.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(i, ty)| {
            (1 << i) & requirements.memory_type_bits != 0 && ty.property_flags & memory == memory
        })
        .map(|(i, _)| i as _)
        .ok_or(Error::Creation)
}
