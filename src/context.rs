use crate::device::DeviceInner;
use crate::memory;
use crate::prelude::*;

use std::borrow;
use std::ffi;
use std::mem;
use std::os::raw;
use std::ptr;
use std::sync::{Arc, Mutex};

use ash::extensions::{ext, khr};
use ash::{vk, Entry, Instance};

#[cfg(target_os = "android")]
use raw_window_handle::AndroidDisplayHandle;
#[cfg(target_os = "windows")]
use raw_window_handle::WindowsDisplayHandle;
#[cfg(target_os = "linux")]
use raw_window_handle::XlibDisplayHandle;

use semver::Version;

use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle,
};

const API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);

pub(crate) const SPECIAL_IMAGE_BINDING: u32 = 2;
pub(crate) const SPECIAL_BUFFER_BINDING: u32 = 3;
pub(crate) const DEVICE_ADDRESS_BUFFER_BINDING: u32 = 4;

pub(crate) const DESCRIPTOR_COUNT: u32 = 200;

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        borrow::Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        borrow::Cow::from("")
    } else {
        ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

#[allow(dead_code)]
#[derive(Clone)]
///The context is your gateway to boson. It starts the vulkan backend and roughly translates to the vulkan instance.
pub struct Context {
    inner: Arc<ContextInner>,
}

pub struct ContextInner {
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    debug: Option<(ext::DebugUtils, vk::DebugUtilsMessengerEXT)>,
}

pub struct ContextInfo<'a> {
    pub enable_validation: bool,
    pub application_name: &'a str,
    pub application_version: Version,
    pub engine_name: &'a str,
    pub engine_version: Version,
    pub display: RawDisplayHandle,
}

impl Default for ContextInfo<'_> {
    fn default() -> Self {
        Self {
            enable_validation: false,
            application_name: "",
            application_version: Version::new(0, 1, 0),
            engine_name: "",
            engine_version: Version::new(0, 1, 0),
            #[cfg(target_os = "windows")]
            display: RawDisplayHandle::Windows(WindowsDisplayHandle::empty()),
            #[cfg(target_os = "linux")]
            display: RawDisplayHandle::Xlib(XlibDisplayHandle::empty()),
            #[cfg(target_os = "android")]
            display: RawDisplayHandle::Android(AndroidDisplayHandle::empty()),
        }
    }
}

impl Context {
    ///Create a new context and start the backend Vulkan instance.
    pub fn new(info: ContextInfo<'_>) -> Result<Self> {
        let entry = Entry::linked();

        let application_name =
            ffi::CString::new(info.application_name).map_err(|_| Error::Creation)?;

        let application_version = vk::make_api_version(
            0,
            info.application_version.major as u32,
            info.application_version.minor as u32,
            info.application_version.patch as u32,
        );

        let engine_name = ffi::CString::new(info.engine_name).map_err(|_| Error::Creation)?;

        let engine_version = vk::make_api_version(
            0,
            info.engine_version.major as u32,
            info.engine_version.minor as u32,
            info.engine_version.patch as u32,
        );

        let application_info = {
            let p_application_name = application_name.as_c_str().as_ptr();
            let p_engine_name = engine_name.as_c_str().as_ptr();

            vk::ApplicationInfo {
                api_version: API_VERSION,
                p_application_name,
                application_version,
                p_engine_name,
                engine_version,
                ..Default::default()
            }
        };

        //SAFETY String is correct
        let mut layers = vec![];

        if info.enable_validation {
            layers.push(unsafe {
                ffi::CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0")
            });
        }

        let mut extensions = vec![];

        #[cfg(not(target_os = "android"))]
        extensions.push(ext::DebugUtils::name());

        let surface_extension_names =
            ash_window::enumerate_required_extensions(info.display).expect("Unsupported Surface");

        let p_application_info = &application_info;

        let enabled_layer_names = layers.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        let enabled_layer_count = enabled_layer_names.len() as u32;

        let enabled_extension_names = extensions
            .iter()
            .map(|s| s.as_ptr())
            .chain(surface_extension_names.iter().copied())
            .collect::<Vec<_>>();
        let enabled_extension_count = enabled_extension_names.len() as u32;

        let instance_create_info = {
            let pp_enabled_layer_names = enabled_layer_names.as_ptr();
            let pp_enabled_extension_names = enabled_extension_names.as_ptr();

            vk::InstanceCreateInfo {
                p_application_info,
                enabled_layer_count,
                pp_enabled_layer_names,
                enabled_extension_count,
                pp_enabled_extension_names,
                ..Default::default()
            }
        };

        //SAFETY this is correct
        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .map_err(|_| Error::Creation)?
        };

        let debug_utils_messenger_create_info = if info.enable_validation {
            Some(vk::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(debug_callback),
                ..Default::default()
            })
        } else {
            None
        };

        let debug = if let Some(info) = debug_utils_messenger_create_info {
            let loader = ext::DebugUtils::new(&entry, &instance);
            //SAFETY this is correct, idk what else to say lol
            let callback = unsafe {
                loader
                    .create_debug_utils_messenger(&info, None)
                    .map_err(|_| Error::Creation)?
            };
            Some((loader, callback))
        } else {
            None
        };

        Ok(Self {
            inner: Arc::new(ContextInner {
                entry,
                instance,
                debug,
            }),
        })
    }

    ///Creates the device, a handle to the system GPU which allows you to interact and manage the graphics pipeline.
    pub fn create_device(&self, info: DeviceInfo) -> Result<Device> {
        let ContextInner {
            entry, instance, ..
        } = &*self.inner;

        let surface_loader = khr::Surface::new(&entry, &instance);
        let surface_handle = unsafe {
            ash_window::create_surface(&entry, &instance, info.display, info.window, None)
        }
        .map_err(|_| Error::CreateSurface)?;

        //SAFETY instance is initialized
        let mut physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|_| Error::EnumeratePhysicalDevices)?
            .into_iter()
            .filter_map(|physical_device| {
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
                    .into_iter()
                    .enumerate()
                    .find_map(|(index, info)| {
                        let graphics_support = info.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                        let compute_support = info.queue_flags.contains(vk::QueueFlags::COMPUTE);
                        let surface_support = unsafe {
                            surface_loader.get_physical_device_surface_support(
                                physical_device,
                                index as u32,
                                surface_handle,
                            )
                        }
                        .ok()?;

                        if graphics_support && compute_support && surface_support {
                            Some((physical_device, index))
                        } else {
                            None
                        }
                    })
            })
            .map(|(physical_device, index)| {
                let properties =
                    unsafe { instance.get_physical_device_properties(physical_device) }.into();
                let features =
                    unsafe { instance.get_physical_device_features(physical_device) }.into();

                let details = crate::device::Details {
                    properties,
                    features,
                };

                let selector = info.selector;

                let score = selector(details);

                (score, physical_device, index)
            })
            .collect::<Vec<_>>();

        physical_devices.sort_by(|(a, _, _), (b, _, _)| a.cmp(b));

        let Some((_, physical_device, queue_family_index)) = physical_devices.pop() else {
            panic!("no suitable device found");
        };

        let queue_family_index = queue_family_index as u32;

        let queue_family_indices = vec![queue_family_index];

        let mut layers = vec![];

        if self.inner.debug.is_some() {
            layers.push(unsafe {
                ffi::CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0")
            });
        }

        let extensions = [khr::Swapchain::name()];

        let mut robustness2_features = {
            vk::PhysicalDeviceRobustness2FeaturesEXT {
                null_descriptor: true as _,
                ..Default::default()
            }
        };

        let mut synchronization2_features = {
            let p_next = &mut robustness2_features as *mut _ as *mut _;

            vk::PhysicalDeviceSynchronization2Features {
                synchronization2: true as _,
                ..Default::default()
            }
        };

        let mut scalar_block_layout_features = {
            let p_next = &mut synchronization2_features as *mut _ as *mut _;

            vk::PhysicalDeviceScalarBlockLayoutFeatures {
                p_next,
                scalar_block_layout: true as _,
                ..Default::default()
            }
        };

        let mut buffer_address_features = {
            let p_next = &mut scalar_block_layout_features as *mut _ as *mut _;

            vk::PhysicalDeviceBufferDeviceAddressFeatures {
                p_next,
                buffer_device_address: true as _,
                ..Default::default()
            }
        };

        let mut indexing_features = {
            let p_next = &mut buffer_address_features as *mut _ as *mut _;

            vk::PhysicalDeviceDescriptorIndexingFeatures {
                p_next,
                runtime_descriptor_array: true as _,
                descriptor_binding_partially_bound: true as _,
                descriptor_binding_update_unused_while_pending: true as _,
                descriptor_binding_storage_buffer_update_after_bind: true as _,
                descriptor_binding_storage_image_update_after_bind: true as _,
                descriptor_binding_sampled_image_update_after_bind: true as _,
                ..Default::default()
            }
        };

        let mut dynamic_rendering_features = {
            let p_next = &mut indexing_features as *mut _ as *mut _;

            vk::PhysicalDeviceDynamicRenderingFeatures {
                p_next,
                dynamic_rendering: true as _,
                ..Default::default()
            }
        };

        let mut features = {
            #[cfg(all(feature = "bindless"))]
            let p_next = &mut dynamic_rendering_features as *mut _ as *mut _;

            vk::PhysicalDeviceFeatures2 {
                p_next: &mut synchronization2_features as *mut _ as *mut _,
                features: vk::PhysicalDeviceFeatures {
                    multi_draw_indirect: true as _,
                    ..info.features.into()
                },
                ..Default::default()
            }
        };

        let priorities = [1.0];

        let device_queue_create_infos = [{
            let queue_count = 1;
            let p_queue_priorities = priorities.as_ptr();

            vk::DeviceQueueCreateInfo {
                queue_family_index,
                queue_count,
                p_queue_priorities,
                ..Default::default()
            }
        }];

        let enabled_layer_names = layers.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        let enabled_layer_count = enabled_layer_names.len() as u32;

        let enabled_extension_names = extensions.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        let enabled_extension_count = enabled_extension_names.len() as u32;

        let mut maintenance4_features = {
            vk::PhysicalDeviceMaintenance4Features {
                p_next: &mut features as *mut _ as *mut _,
                maintenance4: true as _,
                ..Default::default()
            }
        };

        let device_create_info = {
            #[cfg(all(feature = "bindless"))]
            let p_next = &mut features as *mut _ as *mut _;
            #[cfg(not(feature = "bindless"))]
            let p_next = &mut maintenance4_features as *mut _ as *mut _;

            let queue_create_info_count = device_queue_create_infos.len() as _;
            let p_queue_create_infos = device_queue_create_infos.as_ptr();

            let pp_enabled_layer_names = enabled_layer_names.as_ptr();
            let pp_enabled_extension_names = enabled_extension_names.as_ptr();

            vk::DeviceCreateInfo {
                p_next,
                queue_create_info_count,
                p_queue_create_infos,
                enabled_layer_count,
                pp_enabled_layer_names,
                enabled_extension_count,
                pp_enabled_extension_names,
                ..Default::default()
            }
        };

        let logical_device = unsafe {
            self.inner
                .instance
                .create_device(physical_device, &device_create_info, None)
        }
        .map_err(|_| Error::CreateLogicalDevice)?;

        #[cfg(all(feature = "bindless"))]
        let bindless = {
            let descriptor_set_layout_bindings = [
                vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    descriptor_count: DESCRIPTOR_COUNT,
                    stage_flags: vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: DESCRIPTOR_COUNT,
                    stage_flags: vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: SPECIAL_IMAGE_BINDING,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: DESCRIPTOR_COUNT,
                    stage_flags: vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: SPECIAL_BUFFER_BINDING,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: DESCRIPTOR_COUNT,
                    stage_flags: vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: DEVICE_ADDRESS_BUFFER_BINDING,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
            ];

            let binding_flags = [vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND; 5];

            let descriptor_set_layout_binding_flags_create_info = {
                let binding_count = binding_flags.len() as u32;

                let p_binding_flags = binding_flags.as_ptr();

                vk::DescriptorSetLayoutBindingFlagsCreateInfo {
                    binding_count,
                    p_binding_flags,
                    ..Default::default()
                }
            };

            let descriptor_set_layout_create_info = {
                let p_next =
                    &descriptor_set_layout_binding_flags_create_info as *const _ as *const _;

                let flags = vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;

                let binding_count = descriptor_set_layout_bindings.len() as u32;

                let p_bindings = descriptor_set_layout_bindings.as_ptr();

                vk::DescriptorSetLayoutCreateInfo {
                    p_next,
                    flags,
                    binding_count,
                    p_bindings,
                    ..Default::default()
                }
            };

            let descriptor_set_layout = unsafe {
                logical_device
                    .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
            }
            .map_err(|_| Error::CreateDescriptorSetLayout)?;

            let set_layouts = [descriptor_set_layout];

            let descriptor_set_allocate_info = {
                let descriptor_set_count = 1;

                let p_set_layouts = set_layouts.as_ptr();

                vk::DescriptorSetAllocateInfo {
                    descriptor_pool,
                    descriptor_set_count,
                    p_set_layouts,
                    ..Default::default()
                }
            };

            let descriptor_set =
                unsafe { logical_device.allocate_descriptor_sets(&descriptor_set_allocate_info) }
                    .map_err(|_| Error::AllocateDescriptorSets)?[0];

            let allocation_size = (DESCRIPTOR_COUNT * mem::size_of::<u64>() as u32) as u64;

            let staging_address_buffer = {
                let buffer_create_info = vk::BufferCreateInfo {
                    size: allocation_size as _,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                };

                unsafe { logical_device.create_buffer(&buffer_create_info, None) }
                    .map_err(|_| Error::CreateBuffer)?
            };

            let staging_address_memory = {
                let memory_requirements = unsafe {
                    logical_device.get_buffer_memory_requirements(staging_address_buffer)
                };

                let memory_properties =
                    unsafe { instance.get_physical_device_memory_properties(physical_device) };

                let memory_type_index = memory::type_index(
                    &memory_requirements,
                    &memory_properties,
                    Memory::HOST_ACCESS.into(),
                )?;

                let memory_allocate_info = vk::MemoryAllocateInfo {
                    allocation_size,
                    memory_type_index,
                    ..Default::default()
                };

                unsafe { logical_device.allocate_memory(&memory_allocate_info, None) }
                    .map_err(|_| Error::AllocateMemory)?
            };

            unsafe {
                logical_device.bind_buffer_memory(staging_address_buffer, staging_address_memory, 0)
            }
            .map_err(|_| Error::BindBufferMemory)?;

            let general_address_buffer = {
                let buffer_create_info = vk::BufferCreateInfo {
                    size: allocation_size as _,
                    usage: vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                };

                unsafe { logical_device.create_buffer(&buffer_create_info, None) }
                    .map_err(|_| Error::CreateBuffer)?
            };

            let general_address_memory = {
                let memory_requirements = unsafe {
                    logical_device.get_buffer_memory_requirements(general_address_buffer)
                };

                let memory_properties =
                    unsafe { instance.get_physical_device_memory_properties(physical_device) };

                let memory_type_index = memory::type_index(
                    &memory_requirements,
                    &memory_properties,
                    Memory::empty().into(),
                )?;

                let memory_allocate_info = vk::MemoryAllocateInfo {
                    allocation_size,
                    memory_type_index,
                    ..Default::default()
                };

                unsafe { logical_device.allocate_memory(&memory_allocate_info, None) }
                    .map_err(|_| Error::AllocateMemory)?
            };

            unsafe {
                logical_device.bind_buffer_memory(general_address_buffer, general_address_memory, 0)
            }
            .map_err(|_| Error::BindBufferMemory)?;

            Bindless {
                descriptor_set,
                descriptor_set_layout,
                staging_address_buffer,
                staging_address_memory,
                general_address_buffer,
                general_address_memory,
            }
        };

        let command_pool_create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index,
            ..Default::default()
        };

        let command_pool =
            unsafe { logical_device.create_command_pool(&command_pool_create_info, None) }
                .map_err(|_| Error::CreateCommandPool)?;

        let resources = Mutex::new(DeviceResources::new());

        Ok(Device {
            inner: Arc::new(DeviceInner {
                #[cfg(all(feature = "bindless"))]
                bindless,
                context: self.inner.clone(),
                surface: (surface_loader, surface_handle),
                physical_device,
                logical_device,
                queue_family_indices,
                resources,
                command_pool,
            }),
        })
    }
}
