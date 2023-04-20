use crate::prelude::*;

use ash::vk;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinarySemaphore(pub(crate) u32);

impl From<BinarySemaphore> for u32 {
    fn from(handle: BinarySemaphore) -> Self {
        handle.0
    }
}

impl From<u32> for BinarySemaphore {
    fn from(handle: u32) -> Self {
        Self(handle)
    }
}

pub struct InternalSemaphore {
    pub(crate) semaphores: Vec<vk::Semaphore>,
    pub(crate) debug_name: String,
}

pub struct BinarySemaphoreInfo<'a> {
    pub debug_name: &'a str,
}

impl Default for BinarySemaphoreInfo<'_> {
    fn default() -> Self {
        Self {
            debug_name: "Binary Semaphore",
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct TimelineSemaphore(pub(crate) u32);

impl From<TimelineSemaphore> for u32 {
    fn from(handle: TimelineSemaphore) -> Self {
        handle.0
    }
}

impl From<u32> for TimelineSemaphore {
    fn from(handle: u32) -> Self {
        Self(handle)
    }
}

pub struct TimelineSemaphoreInfo<'a> {
    pub initial_value: u64,
    pub debug_name: &'a str,
}

impl Default for TimelineSemaphoreInfo<'_> {
    fn default() -> Self {
        Self {
            initial_value: 0,
            debug_name: "Timeline Semaphore",
        }
    }
}
