use crate::context::ContextInner;
use crate::memory;
use crate::pipeline::PipelineCompilerInner;
use crate::prelude::*;
use crate::semaphore::InternalSemaphore;

use std::default::default;
use std::ffi;
use std::marker;
use std::mem;
use std::ops;
use std::os::raw;
use std::sync::{Arc, Mutex};

use ash::extensions::{ext, khr};
use ash::{vk, Entry, Instance};

use bitflags::bitflags;

use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle,
};

#[cfg(target_os = "windows")]
use raw_window_handle::{Win32WindowHandle, WindowsDisplayHandle};

#[cfg(target_os = "linux")]
use raw_window_handle::{XlibDisplayHandle, XlibWindowHandle};

pub(crate) const MAX_FRAMES_IN_FLIGHT: usize = 3;

pub fn default_device_selector(details: Details) -> usize {
    let mut score = 0;

    score += match details.properties.device_type {
        DeviceType::Discrete => 10000,
        DeviceType::Virtual => 1000,
        DeviceType::Integrated => 100,
        _ => 0,
    };

    score += details.properties.limits.max_memory_allocation_count as usize / 1000;
    score += details.properties.limits.max_descriptor_set_storage_buffers as usize / 1000;
    score += details.properties.limits.max_image_array_layers as usize / 1000;

    score
}

pub trait DeviceSelector = ops::Fn(Details) -> usize;

pub(crate) struct DeviceResource<T, U: Into<u32> + From<u32> + Copy> {
    reprs: Vec<Option<T>>,
    available: Vec<u32>,
    marker: marker::PhantomData<U>,
}

impl<T, U: Into<u32> + From<u32> + Copy> DeviceResource<T, U> {
    pub fn new() -> Self {
        Self {
            reprs: vec![],
            available: vec![],
            marker: marker::PhantomData,
        }
    }

    pub fn add(&mut self, repr: T) -> U {
        if self.available.len() > 0 {
            let index = self.available.pop().unwrap();
            self.reprs.insert(index as usize, Some(repr));
            index
        } else {
            let index = self.reprs.len();
            self.reprs.push(Some(repr));
            index as u32
        }
        .into()
    }

    pub fn count(&self) -> usize {
        self.reprs.len()
    }

    pub fn get(&self, handle: U) -> Option<&'_ T> {
        if let Some(repr) = self.reprs.get(handle.into() as usize) {
            return repr.as_ref();
        }

        None
    }

    pub fn get_mut(&mut self, handle: U) -> Option<&'_ mut T> {
        if let Some(repr) = self.reprs.get_mut(handle.into() as usize) {
            return repr.as_mut();
        }

        None
    }

    pub fn remove(&mut self, handle: U) -> Option<T> {
        let index = handle.into();

        if self.reprs.len() < index as usize {
            return None;
        }

        if let Some(repr) = mem::replace(&mut self.reprs[index as usize], None) {
            self.available.push(index);
            return Some(repr);
        }

        None
    }
}

pub(crate) struct DeviceResources {
    pub(crate) buffers: DeviceResource<InternalBuffer, Buffer>,
    pub(crate) images: DeviceResource<InternalImage, Image>,
    pub(crate) swapchains: DeviceResource<InternalSwapchain, Swapchain>,
    pub(crate) binary_semaphores: DeviceResource<InternalSemaphore, BinarySemaphore>,
    pub(crate) timeline_semaphores: DeviceResource<InternalSemaphore, TimelineSemaphore>,
}

impl DeviceResources {
    pub fn new() -> Self {
        Self {
            buffers: DeviceResource::<InternalBuffer, Buffer>::new(),
            images: DeviceResource::<InternalImage, Image>::new(),
            swapchains: DeviceResource::<InternalSwapchain, Swapchain>::new(),
            binary_semaphores: DeviceResource::<InternalSemaphore, BinarySemaphore>::new(),
            timeline_semaphores: DeviceResource::<InternalSemaphore, TimelineSemaphore>::new(),
        }
    }
}

pub struct Device {
    pub(crate) inner: Arc<DeviceInner>,
}

pub struct DeviceInner {
    pub(crate) context: Arc<ContextInner>,
    pub(crate) resources: Mutex<DeviceResources>,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) logical_device: ash::Device,
    pub(crate) surface: (khr::Surface, vk::SurfaceKHR),
    pub(crate) queue_family_indices: Vec<u32>,
    pub(crate) descriptor_pool: vk::DescriptorPool,
    pub(crate) descriptor_set: vk::DescriptorSet,
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) general_address_buffer: vk::Buffer,
    pub(crate) general_address_memory: vk::DeviceMemory,
    pub(crate) staging_address_buffer: vk::Buffer,
    pub(crate) staging_address_memory: vk::DeviceMemory,
}

pub struct DeviceInfo<'a> {
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
    pub selector: &'a dyn DeviceSelector,
    pub features: Features,
    pub debug_name: &'a str,
}

impl Default for DeviceInfo<'_> {
    fn default() -> Self {
        Self {
            #[cfg(target_os = "windows")]
            display: RawDisplayHandle::Windows(WindowsDisplayHandle::empty()),
            #[cfg(target_os = "windows")]
            window: RawWindowHandle::Win32(Win32WindowHandle::empty()),
            #[cfg(target_os = "linux")]
            display: RawDisplayHandle::Xlib(XlibDisplayHandle::empty()),
            #[cfg(target_os = "linux")]
            window: RawWindowHandle::Xlib(XlibWindowHandle::empty()),
            selector: &default_device_selector,
            features: Default::default(),
            debug_name: "Device",
        }
    }
}

bitflags! {
    pub struct SampleCount: u32 {
        const TYPE_1 = 0x0000_0001;
        const TYPE_2 = 0x0000_0010;
        const TYPE_4 = 0x0000_0100;
        const TYPE_8 = 0x0000_1000;
        const TYPE_16 = 0x0001_0000;
        const TYPE_32 = 0x0010_0000;
        const TYPE_64 = 0x0100_0000;
    }
}

pub struct Details {
    pub properties: Properties,
    pub features: Features,
}

pub struct Properties {
    pub device_type: DeviceType,
    pub limits: Limits,
}

impl From<vk::PhysicalDeviceProperties> for Properties {
    fn from(properties: vk::PhysicalDeviceProperties) -> Self {
        Self {
            device_type: properties.device_type.into(),
            limits: properties.limits.into(),
        }
    }
}

#[derive(Default)]
pub struct Features {
    pub robust_buffer_access: bool,
    pub full_draw_index_uint32: bool,
    pub image_cube_array: bool,
    pub independent_blend: bool,
    pub geometry_shader: bool,
    pub tessellation_shader: bool,
    pub sample_rate_shading: bool,
    pub dual_src_blend: bool,
    pub logic_op: bool,
    pub multi_draw_indirect: bool,
    pub draw_indirect_first_instance: bool,
    pub depth_clamp: bool,
    pub depth_bias_clamp: bool,
    pub fill_mode_non_solid: bool,
    pub depth_bounds: bool,
    pub wide_lines: bool,
    pub large_points: bool,
    pub alpha_to_one: bool,
    pub multi_viewport: bool,
    pub sampler_anisotropy: bool,
    pub texture_compression_etc2: bool,
    pub texture_compression_astc_ldr: bool,
    pub texture_compression_bc: bool,
    pub occlusion_query_precise: bool,
    pub pipeline_statistics_query: bool,
    pub vertex_pipeline_stores_and_atomics: bool,
    pub fragment_stores_and_atomics: bool,
    pub shader_tessellation_and_geometry_point_size: bool,
    pub shader_image_gather_extended: bool,
    pub shader_storage_image_extended_formats: bool,
    pub shader_storage_image_multisample: bool,
    pub shader_storage_image_read_without_format: bool,
    pub shader_storage_image_write_without_format: bool,
    pub shader_uniform_buffer_array_dynamic_indexing: bool,
    pub shader_sampled_image_array_dynamic_indexing: bool,
    pub shader_storage_buffer_array_dynamic_indexing: bool,
    pub shader_storage_image_array_dynamic_indexing: bool,
    pub shader_clip_distance: bool,
    pub shader_cull_distance: bool,
    pub shader_float64: bool,
    pub shader_int64: bool,
    pub shader_int16: bool,
    pub shader_resource_residency: bool,
    pub shader_resource_min_lod: bool,
    pub sparse_binding: bool,
    pub sparse_residency_buffer: bool,
    pub sparse_residency_image2_d: bool,
    pub sparse_residency_image3_d: bool,
    pub sparse_residency2_samples: bool,
    pub sparse_residency4_samples: bool,
    pub sparse_residency8_samples: bool,
    pub sparse_residency16_samples: bool,
    pub sparse_residency_aliased: bool,
    pub variable_multisample_rate: bool,
    pub inherited_queries: bool,
}

impl From<Features> for vk::PhysicalDeviceFeatures {
    fn from(features: Features) -> Self {
        Self {
            robust_buffer_access: features.robust_buffer_access as _,
            full_draw_index_uint32: features.full_draw_index_uint32 as _,
            image_cube_array: features.image_cube_array as _,
            independent_blend: features.independent_blend as _,
            geometry_shader: features.geometry_shader as _,
            tessellation_shader: features.tessellation_shader as _,
            sample_rate_shading: features.sample_rate_shading as _,
            dual_src_blend: features.dual_src_blend as _,
            logic_op: features.logic_op as _,
            multi_draw_indirect: features.multi_draw_indirect as _,
            draw_indirect_first_instance: features.draw_indirect_first_instance as _,
            depth_clamp: features.depth_clamp as _,
            depth_bias_clamp: features.depth_bias_clamp as _,
            fill_mode_non_solid: features.fill_mode_non_solid as _,
            depth_bounds: features.depth_bounds as _,
            wide_lines: features.wide_lines as _,
            large_points: features.large_points as _,
            alpha_to_one: features.alpha_to_one as _,
            multi_viewport: features.multi_viewport as _,
            sampler_anisotropy: features.sampler_anisotropy as _,
            texture_compression_etc2: features.texture_compression_etc2 as _,
            texture_compression_astc_ldr: features.texture_compression_astc_ldr as _,
            texture_compression_bc: features.texture_compression_bc as _,
            occlusion_query_precise: features.occlusion_query_precise as _,
            pipeline_statistics_query: features.pipeline_statistics_query as _,
            vertex_pipeline_stores_and_atomics: features.vertex_pipeline_stores_and_atomics as _,
            fragment_stores_and_atomics: features.fragment_stores_and_atomics as _,
            shader_tessellation_and_geometry_point_size: features
                .shader_tessellation_and_geometry_point_size
                as _,
            shader_image_gather_extended: features.shader_image_gather_extended as _,
            shader_storage_image_extended_formats: features.shader_storage_image_extended_formats
                as _,
            shader_storage_image_multisample: features.shader_storage_image_multisample as _,
            shader_storage_image_read_without_format: features
                .shader_storage_image_read_without_format
                as _,
            shader_storage_image_write_without_format: features
                .shader_storage_image_write_without_format
                as _,
            shader_uniform_buffer_array_dynamic_indexing: features
                .shader_uniform_buffer_array_dynamic_indexing
                as _,
            shader_sampled_image_array_dynamic_indexing: features
                .shader_sampled_image_array_dynamic_indexing
                as _,
            shader_storage_buffer_array_dynamic_indexing: features
                .shader_storage_buffer_array_dynamic_indexing
                as _,
            shader_storage_image_array_dynamic_indexing: features
                .shader_storage_image_array_dynamic_indexing
                as _,
            shader_clip_distance: features.shader_clip_distance as _,
            shader_cull_distance: features.shader_cull_distance as _,
            shader_float64: features.shader_float64 as _,
            shader_int64: features.shader_int64 as _,
            shader_int16: features.shader_int16 as _,
            shader_resource_residency: features.shader_resource_residency as _,
            shader_resource_min_lod: features.shader_resource_min_lod as _,
            sparse_binding: features.sparse_binding as _,
            sparse_residency_buffer: features.sparse_residency_buffer as _,
            sparse_residency_image2_d: features.sparse_residency_image2_d as _,
            sparse_residency_image3_d: features.sparse_residency_image3_d as _,
            sparse_residency2_samples: features.sparse_residency2_samples as _,
            sparse_residency4_samples: features.sparse_residency4_samples as _,
            sparse_residency8_samples: features.sparse_residency8_samples as _,
            sparse_residency16_samples: features.sparse_residency16_samples as _,
            sparse_residency_aliased: features.sparse_residency_aliased as _,
            variable_multisample_rate: features.variable_multisample_rate as _,
            inherited_queries: features.inherited_queries as _,
        }
    }
}

impl From<vk::PhysicalDeviceFeatures> for Features {
    fn from(features: vk::PhysicalDeviceFeatures) -> Self {
        Self {
            robust_buffer_access: features.robust_buffer_access != 0,
            full_draw_index_uint32: features.full_draw_index_uint32 != 0,
            image_cube_array: features.image_cube_array != 0,
            independent_blend: features.independent_blend != 0,
            geometry_shader: features.geometry_shader != 0,
            tessellation_shader: features.tessellation_shader != 0,
            sample_rate_shading: features.sample_rate_shading != 0,
            dual_src_blend: features.dual_src_blend != 0,
            logic_op: features.logic_op != 0,
            multi_draw_indirect: features.multi_draw_indirect != 0,
            draw_indirect_first_instance: features.draw_indirect_first_instance != 0,
            depth_clamp: features.depth_clamp != 0,
            depth_bias_clamp: features.depth_bias_clamp != 0,
            fill_mode_non_solid: features.fill_mode_non_solid != 0,
            depth_bounds: features.depth_bounds != 0,
            wide_lines: features.wide_lines != 0,
            large_points: features.large_points != 0,
            alpha_to_one: features.alpha_to_one != 0,
            multi_viewport: features.multi_viewport != 0,
            sampler_anisotropy: features.sampler_anisotropy != 0,
            texture_compression_etc2: features.texture_compression_etc2 != 0,
            texture_compression_astc_ldr: features.texture_compression_astc_ldr != 0,
            texture_compression_bc: features.texture_compression_bc != 0,
            occlusion_query_precise: features.occlusion_query_precise != 0,
            pipeline_statistics_query: features.pipeline_statistics_query != 0,
            vertex_pipeline_stores_and_atomics: features.vertex_pipeline_stores_and_atomics != 0,
            fragment_stores_and_atomics: features.fragment_stores_and_atomics != 0,
            shader_tessellation_and_geometry_point_size: features
                .shader_tessellation_and_geometry_point_size
                != 0,
            shader_image_gather_extended: features.shader_image_gather_extended != 0,
            shader_storage_image_extended_formats: features.shader_storage_image_extended_formats
                != 0,
            shader_storage_image_multisample: features.shader_storage_image_multisample != 0,
            shader_storage_image_read_without_format: features
                .shader_storage_image_read_without_format
                != 0,
            shader_storage_image_write_without_format: features
                .shader_storage_image_write_without_format
                != 0,
            shader_uniform_buffer_array_dynamic_indexing: features
                .shader_uniform_buffer_array_dynamic_indexing
                != 0,
            shader_sampled_image_array_dynamic_indexing: features
                .shader_sampled_image_array_dynamic_indexing
                != 0,
            shader_storage_buffer_array_dynamic_indexing: features
                .shader_storage_buffer_array_dynamic_indexing
                != 0,
            shader_storage_image_array_dynamic_indexing: features
                .shader_storage_image_array_dynamic_indexing
                != 0,
            shader_clip_distance: features.shader_clip_distance != 0,
            shader_cull_distance: features.shader_cull_distance != 0,
            shader_float64: features.shader_float64 != 0,
            shader_int64: features.shader_int64 != 0,
            shader_int16: features.shader_int16 != 0,
            shader_resource_residency: features.shader_resource_residency != 0,
            shader_resource_min_lod: features.shader_resource_min_lod != 0,
            sparse_binding: features.sparse_binding != 0,
            sparse_residency_buffer: features.sparse_residency_buffer != 0,
            sparse_residency_image2_d: features.sparse_residency_image2_d != 0,
            sparse_residency_image3_d: features.sparse_residency_image3_d != 0,
            sparse_residency2_samples: features.sparse_residency2_samples != 0,
            sparse_residency4_samples: features.sparse_residency4_samples != 0,
            sparse_residency8_samples: features.sparse_residency8_samples != 0,
            sparse_residency16_samples: features.sparse_residency16_samples != 0,
            sparse_residency_aliased: features.sparse_residency_aliased != 0,
            variable_multisample_rate: features.variable_multisample_rate != 0,
            inherited_queries: features.inherited_queries != 0,
        }
    }
}

pub enum DeviceType {
    Other,
    Integrated,
    Virtual,
    Discrete,
}

impl From<vk::PhysicalDeviceType> for DeviceType {
    fn from(ty: vk::PhysicalDeviceType) -> Self {
        match ty {
            vk::PhysicalDeviceType::INTEGRATED_GPU => Self::Integrated,
            vk::PhysicalDeviceType::VIRTUAL_GPU => Self::Virtual,
            vk::PhysicalDeviceType::DISCRETE_GPU => Self::Discrete,
            _ => Self::Other,
        }
    }
}

pub struct Limits {
    pub max_image_dimension1_d: u32,
    pub max_image_dimension2_d: u32,
    pub max_image_dimension3_d: u32,
    pub max_image_dimension_cube: u32,
    pub max_image_array_layers: u32,
    pub max_texel_buffer_elements: u32,
    pub max_uniform_buffer_range: u32,
    pub max_storage_buffer_range: u32,
    pub max_push_constants_size: u32,
    pub max_memory_allocation_count: u32,
    pub max_sampler_allocation_count: u32,
    pub buffer_image_granularity: u64,
    pub sparse_address_space_size: u64,
    pub max_bound_descriptor_sets: u32,
    pub max_per_stage_descriptor_samplers: u32,
    pub max_per_stage_descriptor_uniform_buffers: u32,
    pub max_per_stage_descriptor_storage_buffers: u32,
    pub max_per_stage_descriptor_sampled_images: u32,
    pub max_per_stage_descriptor_storage_images: u32,
    pub max_per_stage_descriptor_input_attachments: u32,
    pub max_per_stage_resources: u32,
    pub max_descriptor_set_samplers: u32,
    pub max_descriptor_set_uniform_buffers: u32,
    pub max_descriptor_set_uniform_buffers_dynamic: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_descriptor_set_storage_buffers_dynamic: u32,
    pub max_descriptor_set_sampled_images: u32,
    pub max_descriptor_set_storage_images: u32,
    pub max_descriptor_set_input_attachments: u32,
    pub max_vertex_input_attributes: u32,
    pub max_vertex_input_bindings: u32,
    pub max_vertex_input_attribute_offset: u32,
    pub max_vertex_input_binding_stride: u32,
    pub max_vertex_output_components: u32,
    pub max_tessellation_generation_level: u32,
    pub max_tessellation_patch_size: u32,
    pub max_tessellation_control_per_vertex_input_components: u32,
    pub max_tessellation_control_per_vertex_output_components: u32,
    pub max_tessellation_control_per_patch_output_components: u32,
    pub max_tessellation_control_total_output_components: u32,
    pub max_tessellation_evaluation_input_components: u32,
    pub max_tessellation_evaluation_output_components: u32,
    pub max_geometry_shader_invocations: u32,
    pub max_geometry_input_components: u32,
    pub max_geometry_output_components: u32,
    pub max_geometry_output_vertices: u32,
    pub max_geometry_total_output_components: u32,
    pub max_fragment_input_components: u32,
    pub max_fragment_output_attachments: u32,
    pub max_fragment_dual_src_attachments: u32,
    pub max_fragment_combined_output_resources: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size: [u32; 3],
    pub sub_pixel_precision_bits: u32,
    pub sub_texel_precision_bits: u32,
    pub mipmap_precision_bits: u32,
    pub max_draw_indexed_index_value: u32,
    pub max_draw_indirect_count: u32,
    pub max_sampler_lod_bias: f32,
    pub max_sampler_anisotropy: f32,
    pub max_viewports: u32,
    pub max_viewport_dimensions: [u32; 2],
    pub viewport_bounds_range: [f32; 2],
    pub viewport_sub_pixel_bits: u32,
    pub min_memory_map_alignment: usize,
    pub min_texel_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_texel_offset: i32,
    pub max_texel_offset: u32,
    pub min_texel_gather_offset: i32,
    pub max_texel_gather_offset: u32,
    pub min_interpolation_offset: f32,
    pub max_interpolation_offset: f32,
    pub sub_pixel_interpolation_offset_bits: u32,
    pub max_framebuffer_width: u32,
    pub max_framebuffer_height: u32,
    pub max_framebuffer_layers: u32,
    pub framebuffer_color_sample_counts: SampleCount,
    pub framebuffer_depth_sample_counts: SampleCount,
    pub framebuffer_stencil_sample_counts: SampleCount,
    pub framebuffer_no_attachments_sample_counts: SampleCount,
    pub max_color_attachments: u32,
    pub sampled_image_color_sample_counts: SampleCount,
    pub sampled_image_integer_sample_counts: SampleCount,
    pub sampled_image_depth_sample_counts: SampleCount,
    pub sampled_image_stencil_sample_counts: SampleCount,
    pub storage_image_sample_counts: SampleCount,
    pub max_sample_mask_words: u32,
    pub timestamp_compute_and_graphics: bool,
    pub timestamp_period: f32,
    pub max_clip_distances: u32,
    pub max_cull_distances: u32,
    pub max_combined_clip_and_cull_distances: u32,
    pub discrete_queue_priorities: u32,
    pub point_size_range: [f32; 2],
    pub line_width_range: [f32; 2],
    pub point_size_granularity: f32,
    pub line_width_granularity: f32,
    pub strict_lines: bool,
    pub standard_sample_locations: bool,
    pub optimal_buffer_copy_offset_alignment: u64,
    pub optimal_buffer_copy_row_pitch_alignment: u64,
    pub non_coherent_atom_size: u64,
}

impl From<vk::PhysicalDeviceLimits> for Limits {
    fn from(limits: vk::PhysicalDeviceLimits) -> Self {
        Self {
            max_image_dimension1_d: limits.max_image_dimension1_d as _,
            max_image_dimension2_d: limits.max_image_dimension2_d as _,
            max_image_dimension3_d: limits.max_image_dimension3_d as _,
            max_image_dimension_cube: limits.max_image_dimension_cube as _,
            max_image_array_layers: limits.max_image_array_layers as _,
            max_texel_buffer_elements: limits.max_texel_buffer_elements as _,
            max_uniform_buffer_range: limits.max_uniform_buffer_range as _,
            max_storage_buffer_range: limits.max_storage_buffer_range as _,
            max_push_constants_size: limits.max_push_constants_size as _,
            max_memory_allocation_count: limits.max_memory_allocation_count as _,
            max_sampler_allocation_count: limits.max_sampler_allocation_count as _,
            buffer_image_granularity: limits.buffer_image_granularity as _,
            sparse_address_space_size: limits.sparse_address_space_size as _,
            max_bound_descriptor_sets: limits.max_bound_descriptor_sets as _,
            max_per_stage_descriptor_samplers: limits.max_per_stage_descriptor_samplers as _,
            max_per_stage_descriptor_uniform_buffers: limits
                .max_per_stage_descriptor_uniform_buffers
                as _,
            max_per_stage_descriptor_storage_buffers: limits
                .max_per_stage_descriptor_storage_buffers
                as _,
            max_per_stage_descriptor_sampled_images: limits.max_per_stage_descriptor_sampled_images
                as _,
            max_per_stage_descriptor_storage_images: limits.max_per_stage_descriptor_storage_images
                as _,
            max_per_stage_descriptor_input_attachments: limits
                .max_per_stage_descriptor_input_attachments
                as _,
            max_per_stage_resources: limits.max_per_stage_resources as _,
            max_descriptor_set_samplers: limits.max_descriptor_set_samplers as _,
            max_descriptor_set_uniform_buffers: limits.max_descriptor_set_uniform_buffers as _,
            max_descriptor_set_uniform_buffers_dynamic: limits
                .max_descriptor_set_uniform_buffers_dynamic
                as _,
            max_descriptor_set_storage_buffers: limits.max_descriptor_set_storage_buffers as _,
            max_descriptor_set_storage_buffers_dynamic: limits
                .max_descriptor_set_storage_buffers_dynamic
                as _,
            max_descriptor_set_sampled_images: limits.max_descriptor_set_sampled_images as _,
            max_descriptor_set_storage_images: limits.max_descriptor_set_storage_images as _,
            max_descriptor_set_input_attachments: limits.max_descriptor_set_input_attachments as _,
            max_vertex_input_attributes: limits.max_vertex_input_attributes as _,
            max_vertex_input_bindings: limits.max_vertex_input_bindings as _,
            max_vertex_input_attribute_offset: limits.max_vertex_input_attribute_offset as _,
            max_vertex_input_binding_stride: limits.max_vertex_input_binding_stride as _,
            max_vertex_output_components: limits.max_vertex_output_components as _,
            max_tessellation_generation_level: limits.max_tessellation_generation_level as _,
            max_tessellation_patch_size: limits.max_tessellation_patch_size as _,
            max_tessellation_control_per_vertex_input_components: limits
                .max_tessellation_control_per_vertex_input_components
                as _,
            max_tessellation_control_per_vertex_output_components: limits
                .max_tessellation_control_per_vertex_output_components
                as _,
            max_tessellation_control_per_patch_output_components: limits
                .max_tessellation_control_per_patch_output_components
                as _,
            max_tessellation_control_total_output_components: limits
                .max_tessellation_control_total_output_components
                as _,
            max_tessellation_evaluation_input_components: limits
                .max_tessellation_evaluation_input_components
                as _,
            max_tessellation_evaluation_output_components: limits
                .max_tessellation_evaluation_output_components
                as _,
            max_geometry_shader_invocations: limits.max_geometry_shader_invocations as _,
            max_geometry_input_components: limits.max_geometry_input_components as _,
            max_geometry_output_components: limits.max_geometry_output_components as _,
            max_geometry_output_vertices: limits.max_geometry_output_vertices as _,
            max_geometry_total_output_components: limits.max_geometry_total_output_components as _,
            max_fragment_input_components: limits.max_fragment_input_components as _,
            max_fragment_output_attachments: limits.max_fragment_output_attachments as _,
            max_fragment_dual_src_attachments: limits.max_fragment_dual_src_attachments as _,
            max_fragment_combined_output_resources: limits.max_fragment_combined_output_resources
                as _,
            max_compute_shared_memory_size: limits.max_compute_shared_memory_size as _,
            max_compute_work_group_count: limits.max_compute_work_group_count as _,
            max_compute_work_group_invocations: limits.max_compute_work_group_invocations as _,
            max_compute_work_group_size: limits.max_compute_work_group_size as _,
            sub_pixel_precision_bits: limits.sub_pixel_precision_bits as _,
            sub_texel_precision_bits: limits.sub_texel_precision_bits as _,
            mipmap_precision_bits: limits.mipmap_precision_bits as _,
            max_draw_indexed_index_value: limits.max_draw_indexed_index_value as _,
            max_draw_indirect_count: limits.max_draw_indirect_count as _,
            max_sampler_lod_bias: limits.max_sampler_lod_bias as _,
            max_sampler_anisotropy: limits.max_sampler_anisotropy as _,
            max_viewports: limits.max_viewports as _,
            max_viewport_dimensions: limits.max_viewport_dimensions as _,
            viewport_bounds_range: limits.viewport_bounds_range as _,
            viewport_sub_pixel_bits: limits.viewport_sub_pixel_bits as _,
            min_memory_map_alignment: limits.min_memory_map_alignment as _,
            min_texel_buffer_offset_alignment: limits.min_texel_buffer_offset_alignment as _,
            min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment as _,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment as _,
            min_texel_offset: limits.min_texel_offset as _,
            max_texel_offset: limits.max_texel_offset as _,
            min_texel_gather_offset: limits.min_texel_gather_offset as _,
            max_texel_gather_offset: limits.max_texel_gather_offset as _,
            min_interpolation_offset: limits.min_interpolation_offset as _,
            max_interpolation_offset: limits.max_interpolation_offset as _,
            sub_pixel_interpolation_offset_bits: limits.sub_pixel_interpolation_offset_bits as _,
            max_framebuffer_width: limits.max_framebuffer_width as _,
            max_framebuffer_height: limits.max_framebuffer_height as _,
            max_framebuffer_layers: limits.max_framebuffer_layers as _,
            framebuffer_color_sample_counts: SampleCount::from_bits_truncate(
                limits.framebuffer_color_sample_counts.as_raw(),
            ),
            framebuffer_depth_sample_counts: SampleCount::from_bits_truncate(
                limits.framebuffer_depth_sample_counts.as_raw(),
            ),
            framebuffer_stencil_sample_counts: SampleCount::from_bits_truncate(
                limits.framebuffer_stencil_sample_counts.as_raw(),
            ),
            framebuffer_no_attachments_sample_counts: SampleCount::from_bits_truncate(
                limits.framebuffer_no_attachments_sample_counts.as_raw(),
            ),
            max_color_attachments: limits.max_color_attachments as _,
            sampled_image_color_sample_counts: SampleCount::from_bits_truncate(
                limits.sampled_image_color_sample_counts.as_raw(),
            ),
            sampled_image_integer_sample_counts: SampleCount::from_bits_truncate(
                limits.sampled_image_integer_sample_counts.as_raw(),
            ),
            sampled_image_depth_sample_counts: SampleCount::from_bits_truncate(
                limits.sampled_image_depth_sample_counts.as_raw(),
            ),
            sampled_image_stencil_sample_counts: SampleCount::from_bits_truncate(
                limits.sampled_image_stencil_sample_counts.as_raw(),
            ),
            storage_image_sample_counts: SampleCount::from_bits_truncate(
                limits.storage_image_sample_counts.as_raw(),
            ),
            max_sample_mask_words: limits.max_sample_mask_words as _,
            timestamp_compute_and_graphics: limits.timestamp_compute_and_graphics != 0,
            timestamp_period: limits.timestamp_period as _,
            max_clip_distances: limits.max_clip_distances as _,
            max_cull_distances: limits.max_cull_distances as _,
            max_combined_clip_and_cull_distances: limits.max_combined_clip_and_cull_distances as _,
            discrete_queue_priorities: limits.discrete_queue_priorities as _,
            point_size_range: limits.point_size_range as _,
            line_width_range: limits.line_width_range as _,
            point_size_granularity: limits.point_size_granularity as _,
            line_width_granularity: limits.line_width_granularity as _,
            strict_lines: limits.strict_lines != 0,
            standard_sample_locations: limits.standard_sample_locations != 0,
            optimal_buffer_copy_offset_alignment: limits.optimal_buffer_copy_offset_alignment as _,
            optimal_buffer_copy_row_pitch_alignment: limits.optimal_buffer_copy_row_pitch_alignment
                as _,
            non_coherent_atom_size: limits.non_coherent_atom_size as _,
        }
    }
}

impl Device {
    pub fn acquire(&self) -> Image {
        todo!()
    }

    pub fn create_executor<'a>(&self, info: ExecutorInfo<'a>) -> Result<Executor<'a>> {
        let ExecutorInfo {
            optimizer,
            debug_name,
            swapchain,
        } = info;

        let debug_name = debug_name.to_owned();

        let nodes = vec![];

        self.inner
            .resources
            .lock()
            .unwrap()
            .swapchains
            .get(swapchain)
            .ok_or(Error::ResourceNotFound)?;

        Ok(Executor {
            device: self.inner.clone(),
            swapchain,
            optimizer,
            nodes,
            debug_name,
        })
    }

    pub fn create_image(&self, info: ImageInfo<'_>) -> Result<Image> {
        let DeviceInner {
            context,
            physical_device,
            logical_device,
            resources,
            descriptor_set,
            ..
        } = &*self.inner;

        let mut resources = resources.lock().unwrap();

        let ContextInner { instance, .. } = &**context;

        let ImageInfo {
            extent,
            usage,
            format,
            debug_name,
        } = info;

        let (image_type, view_type, extent) = match extent {
            ImageExtent::OneDim(x) => (
                vk::ImageType::TYPE_1D,
                vk::ImageViewType::TYPE_1D,
                vk::Extent3D {
                    width: x as _,
                    height: 1,
                    depth: 1,
                },
            ),
            ImageExtent::TwoDim(x, y) => (
                vk::ImageType::TYPE_2D,
                vk::ImageViewType::TYPE_2D,
                vk::Extent3D {
                    width: x as _,
                    height: y as _,
                    depth: 1,
                },
            ),
            ImageExtent::ThreeDim(x, y, z) => (
                vk::ImageType::TYPE_3D,
                vk::ImageViewType::TYPE_3D,
                vk::Extent3D {
                    width: x as _,
                    height: y as _,
                    depth: z as _,
                },
            ),
        };

        let mut usage = vk::ImageUsageFlags::from(usage);

        if (usage & vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) == vk::ImageUsageFlags::empty() {
            usage |= vk::ImageUsageFlags::STORAGE;
        }

        let image_create_info = vk::ImageCreateInfo {
            image_type,
            extent,
            format: format.into(),
            usage,
            array_layers: 1,
            mip_levels: 1,
            tiling: vk::ImageTiling::OPTIMAL,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            samples: vk::SampleCountFlags::TYPE_1,
            ..default()
        };

        let image = unsafe { logical_device.create_image(&image_create_info, None) }
            .map_err(|_| Error::Creation)?;

        let memory_requirements = unsafe { logical_device.get_image_memory_requirements(image) };

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(*physical_device) };

        let memory_type_index = memory::type_index(
            &memory_requirements,
            &memory_properties,
            Memory::empty().into(),
        )?;

        let allocation_size = memory_requirements.size;

        let memory_allocate_info = {
            vk::MemoryAllocateInfo {
                allocation_size,
                memory_type_index,
                ..default()
            }
        };

        let memory = unsafe { logical_device.allocate_memory(&memory_allocate_info, None) }
            .map_err(|_| Error::Creation)?;

        unsafe { logical_device.bind_image_memory(image, memory, 0) }
            .map_err(|_| Error::Creation)?;

        let image_view_create_info = vk::ImageViewCreateInfo {
            image,
            view_type,
            format: format.into(),
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: format.into(),
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..default()
        };

        let view =
            unsafe { logical_device.create_image_view(&image_view_create_info, None) }.unwrap();

        Ok(resources.images.add(InternalImage::Managed {
            image,
            memory,
            view,
            format,
        }))
    }

    pub fn create_buffer(&self, info: BufferInfo<'_>) -> Result<Buffer> {
        let DeviceInner {
            context,
            physical_device,
            logical_device,
            resources,
            descriptor_set,
            ..
        } = &*self.inner;

        let mut resources = resources.lock().unwrap();

        let ContextInner { instance, .. } = &**context;

        let BufferInfo {
            size,
            usage,
            memory: properties,
            debug_name,
        } = info;

        let size = size as _;

        let allocation_size = size as _;

        let mut usage = usage.into();

        usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let debug_name = debug_name.to_owned();

        let sharing_mode = vk::SharingMode::EXCLUSIVE;

        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode,
            ..default()
        };

        let buffer = unsafe { logical_device.create_buffer(&buffer_create_info, None) }
            .map_err(|_| Error::Creation)?;

        let memory_requirements = unsafe { logical_device.get_buffer_memory_requirements(buffer) };

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(*physical_device) };

        let memory_type_index =
            memory::type_index(&memory_requirements, &memory_properties, properties.into())?;

        let mut memory_allocate_flags_info = vk::MemoryAllocateFlagsInfo {
            flags: vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            ..default()
        };

        let memory_allocate_info = {
            let p_next = &mut memory_allocate_flags_info as *mut _ as *mut _;
            vk::MemoryAllocateInfo {
                p_next,
                allocation_size,
                memory_type_index,
                ..default()
            }
        };

        let memory = unsafe { logical_device.allocate_memory(&memory_allocate_info, None) }
            .map_err(|_| Error::Creation)?;

        unsafe { logical_device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|_| Error::Creation)?;

        let memory = InternalMemory { memory, properties };

        let BufferInfo { size, usage, .. } = info;

        Ok(resources.buffers.add(InternalBuffer {
            buffer,
            memory,
            size,
            usage,
            debug_name,
        }))
    }

    pub fn create_binary_semaphore(
        &self,
        info: BinarySemaphoreInfo<'_>,
    ) -> Result<BinarySemaphore> {
        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*self.inner;

        let BinarySemaphoreInfo { debug_name } = info;

        let debug_name = debug_name.to_owned();

        let mut semaphores = vec![];

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let semaphore = unsafe { logical_device.create_semaphore(&default(), None) }
                .map_err(|_| Error::Creation)?;
            semaphores.push(semaphore);
        }

        Ok(resources
            .lock()
            .unwrap()
            .binary_semaphores
            .add(InternalSemaphore {
                semaphores,
                debug_name,
            }))
    }

    pub fn create_timeline_semaphore(
        &self,
        info: TimelineSemaphoreInfo<'_>,
    ) -> Result<TimelineSemaphore> {
        let DeviceInner {
            logical_device,
            resources,
            ..
        } = &*self.inner;

        let TimelineSemaphoreInfo {
            initial_value,
            debug_name,
            ..
        } = info;

        let debug_name = debug_name.to_owned();

        let semaphore_type = vk::SemaphoreType::TIMELINE;

        let semaphore_type_create_info = vk::SemaphoreTypeCreateInfo {
            initial_value,
            semaphore_type,
            ..default()
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            p_next: &semaphore_type_create_info as *const _ as *const _,
            ..default()
        };

        let mut semaphores = vec![];

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let semaphore = unsafe { logical_device.create_semaphore(&default(), None) }
                .map_err(|_| Error::Creation)?;
            semaphores.push(semaphore);
        }
        Ok(resources
            .lock()
            .unwrap()
            .timeline_semaphores
            .add(InternalSemaphore {
                semaphores,
                debug_name,
            }))
    }

    pub fn acquire_next_image(&self, acquire: Acquire) -> Result<Image> {
        let DeviceInner { resources, .. } = &*self.inner;

        let mut resources = resources.lock().unwrap();

        let semaphores = if let Some(handle) = acquire.semaphore {
            resources
                .binary_semaphores
                .get(handle)
                .ok_or(Error::InvalidResource)?
                .semaphores
                .clone()
        } else {
            vec![]
        };

        let InternalSwapchain {
            loader,
            handle,
            images,
            last_acquisition_index,
            current_frame,
            allow_acquisition,
            ..
        } = resources
            .swapchains
            .get_mut(acquire.swapchain)
            .ok_or(Error::ResourceNotFound)?;

        let semaphore = if semaphores.len() > 0 {
            semaphores[*current_frame]
        } else {
            vk::Semaphore::null()
        };

        if !*allow_acquisition {
            return last_acquisition_index
                .map(|i| images[i as usize])
                .ok_or(Error::FailedToAcquire);
        }

        let (next_image_index, suboptimal) =
            unsafe { loader.acquire_next_image(*handle, u64::MAX, semaphore, vk::Fence::null()) }
                .map_err(|_| Error::FailedToAcquire)?;

        *allow_acquisition = false;
        *last_acquisition_index = Some(next_image_index);

        Ok(images[next_image_index as usize])
    }

    pub fn presentation_format(&self, swapchain: Swapchain) -> Result<Format> {
        let DeviceInner { resources, .. } = &*self.inner;

        let resources = resources.lock().unwrap();

        let InternalSwapchain { format, .. } = resources
            .swapchains
            .get(swapchain)
            .ok_or(Error::ResourceNotFound)?;

        Ok(*format)
    }

    pub fn wait_idle(&self) {
        let DeviceInner { logical_device, .. } = &*self.inner;

        unsafe { logical_device.device_wait_idle() };
    }

    pub fn create_swapchain(&self, info: SwapchainInfo<'_>) -> Result<Swapchain> {
        let DeviceInner {
            context,
            surface: (surface_loader, surface_handle),
            physical_device,
            logical_device,
            queue_family_indices,
            resources,
            ..
        } = &*self.inner;

        let mut resources = resources.lock().unwrap();

        let mut surface_formats = unsafe {
            surface_loader.get_physical_device_surface_formats(*physical_device, *surface_handle)
        }
        .map_err(|_| Error::Creation)?
        .into_iter()
        .filter_map(|surface_format| {
            let selector = info.surface_format_selector;

            let score = selector(surface_format.format.try_into().ok()?);

            Some((score, surface_format))
        })
        .collect::<Vec<_>>();

        surface_formats.sort_by(|(a, _), (b, _)| b.cmp(a));

        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(*physical_device, *surface_handle)
        }
        .map_err(|_| Error::Creation)?;

        let ContextInner { instance, .. } = &**context;

        let swapchain_loader = khr::Swapchain::new(&instance, &logical_device);

        let Some((_, vk::SurfaceFormatKHR { format, color_space })) = surface_formats.pop() else {
                panic!("failed to find suitable surface format");
        };

        let swapchain_create_info = {
            let surface = surface_handle;

            let min_image_count = match info.present_mode {
                PresentMode::TripleBufferWaitForVBlank => 3,
                PresentMode::DoNotWaitForVBlank
                | PresentMode::DoubleBufferWaitForVBlank
                | PresentMode::DoubleBufferWaitForVBlankRelaxed => 2,
            };

            let image_format = format;
            let image_color_space = color_space;

            let image_extent = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D {
                    width: info.width,
                    height: info.height,
                },
                _ => surface_capabilities.current_extent,
            };

            let image_array_layers = 1;
            let image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
            let image_sharing_mode = vk::SharingMode::EXCLUSIVE;

            let queue_family_index_count = queue_family_indices.len() as _;

            let p_queue_family_indices = queue_family_indices.as_ptr();

            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };

            let composite_alpha = vk::CompositeAlphaFlagsKHR::OPAQUE;

            let preferred_present_mode = vk::PresentModeKHR::from(info.present_mode);

            let present_modes = unsafe {
                surface_loader
                    .get_physical_device_surface_present_modes(*physical_device, *surface_handle)
            }
            .map_err(|_| Error::Creation)?;

            let present_mode = present_modes
                .into_iter()
                .find(|&present_mode| present_mode == preferred_present_mode)
                .unwrap_or(vk::PresentModeKHR::IMMEDIATE);

            let clipped = true as _;

            let surface = *surface;

            let old_swapchain = if let Some(swapchain) = info.old_swapchain {
                resources
                    .swapchains
                    .get(swapchain)
                    .ok_or(Error::ResourceNotFound)?
                    .handle
            } else {
                vk::SwapchainKHR::null()
            };

            vk::SwapchainCreateInfoKHR {
                surface,
                min_image_count,
                image_format,
                image_color_space,
                image_extent,
                image_array_layers,
                image_usage,
                image_sharing_mode,
                queue_family_index_count,
                p_queue_family_indices,
                pre_transform,
                composite_alpha,
                present_mode,
                clipped,
                old_swapchain,
                ..Default::default()
            }
        };

        let swapchain_handle =
            unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }
                .map_err(|_| Error::Creation)?;

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain_handle) }
            .map_err(|_| Error::Creation)?;

        let images = swapchain_images
            .into_iter()
            .map(|image| {
                let image_view_create_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..default()
                };

                let image_view =
                    unsafe { logical_device.create_image_view(&image_view_create_info, None) }
                        .unwrap();

                (image, image_view)
            })
            .map(|(image, view)| InternalImage::Swapchain {
                image,
                view,
                format: format.try_into().unwrap(),
            })
            .map(|internal_image| resources.images.add(internal_image))
            .collect::<Vec<_>>();

        let loader = swapchain_loader;

        let handle = swapchain_handle;

        let format = format.try_into().map_err(|_| Error::Creation)?;

        let last_acquisition_index = None;

        let current_frame = 0;

        let allow_acquisition = true;

        Ok(resources.swapchains.add(InternalSwapchain {
            loader,
            handle,
            format,
            images,
            last_acquisition_index,
            current_frame,
            allow_acquisition,
        }))
    }

    pub fn create_pipeline_compiler(&self, info: PipelineCompilerInfo<'_>) -> PipelineCompiler {
        PipelineCompiler {
            inner: Arc::new(PipelineCompilerInner {
                device: self.inner.clone(),
                compiler: info.compiler,
                source_path: info.source_path.to_path_buf(),
                asset_path: info.asset_path.to_path_buf(),
                debug_name: info.debug_name.to_owned(),
            }),
        }
    }
}
