use crate::device::DeviceInner;
use crate::device::MAX_FRAMES_IN_FLIGHT;
use crate::prelude::*;

use core::slice;
use std::borrow;
use std::char::MAX;
use std::env;
use std::ffi;
use std::fmt;
use std::fs;
use std::io::Write;
use std::iter;
use std::mem;
use std::path;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use std::ptr;
use std::sync::{Arc, Mutex};

use ash::vk;

use ash::vk::DescriptorType;
use ash::vk::ShaderStageFlags;
use bitflags::bitflags;

use lazy_static::lazy_static;

pub type Spv = Vec<u32>;

///Requires a name, a type, and the includes
#[derive(Clone)]
pub struct Shader {
    pub ty: ShaderType,
    pub source: Vec<u8>,
    pub defines: Vec<Define>,
}

impl Default for Shader {
    fn default() -> Self {
        Self {
            ty: ShaderType::Vertex,
            source: vec![],
            defines: vec![],
        }
    }
}
#[derive(Clone, Copy, Default)]
pub enum ShaderLanguage {
    #[default]
    Glsl,
    Hlsl,
}

#[derive(Clone, Copy)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
}

impl fmt::Display for ShaderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ShaderType::Vertex => "vertex",
                ShaderType::Fragment => "fragment",
                ShaderType::Compute => "compute",
            }
        )
    }
}

impl From<ShaderType> for vk::ShaderStageFlags {
    fn from(ty: ShaderType) -> Self {
        match ty {
            ShaderType::Vertex => Self::VERTEX,
            ShaderType::Fragment => Self::FRAGMENT,
            ShaderType::Compute => Self::COMPUTE,
        }
    }
}

#[derive(Default)]
pub struct ShaderCompilerInfo {
    language: ShaderLanguage,
}

pub struct ShaderCompilationOptions<'a> {
    include_dir: &'a Path,
    source: &'a [u8],
    ty: ShaderType,
    defines: &'a [Define],
}

#[derive(Clone)]
pub struct Define {
    name: String,
    value: String,
}

pub trait ShaderCompiler: 'static + Send + Sync {
    fn compile_to_spv(&self, options: ShaderCompilationOptions) -> Result<Spv>;
}

pub struct ByteToSpirvCompiler;

impl ShaderCompiler for ByteToSpirvCompiler {
    fn compile_to_spv(&self, options: ShaderCompilationOptions) -> Result<Spv> {
        Ok(unsafe {
            slice::from_raw_parts(
                options.source.as_ptr() as *const _,
                options.source.len() / mem::size_of::<u32>(),
            )
        }
        .to_vec())
    }
}

#[cfg(all(feature = "shaderc"))]
use shaderc::ResolvedInclude;

#[cfg(all(feature = "shaderc"))]
pub struct Shaderc {
    compiler: shaderc::Compiler,
}

#[cfg(all(feature = "shaderc"))]
impl Shaderc {
    fn new() -> Self {
        Self {
            compiler: shaderc::Compiler::new().unwrap(),
        }
    }
}

#[cfg(all(feature = "shaderc"))]
impl ShaderCompiler for Shaderc {
    fn compile_to_spv(&self, options: ShaderCompilationOptions) -> Result<Spv> {
        let source = String::from_utf8(options.source.to_vec()).unwrap();

        let mut additional_options = shaderc::CompileOptions::new().unwrap();

        for Define { name, value } in options.defines {
            additional_options.add_macro_definition(name, Some(value));
        }

        additional_options.add_macro_definition("shader_type_vertex", Some(&0.to_string()));
        additional_options.add_macro_definition("shader_type_fragment", Some(&1.to_string()));
        additional_options.add_macro_definition("shader_type_compute", Some(&2.to_string()));

        additional_options.set_generate_debug_info();
        additional_options.add_macro_definition(
            "shader_type",
            Some(
                &match options.ty {
                    ShaderType::Vertex => 0,
                    ShaderType::Fragment => 1,
                    ShaderType::Compute => 2,
                }
                .to_string(),
            ),
        );

        let include_dir = options.include_dir.to_owned();
        additional_options.set_include_callback(move |name, _, _, _| {
            let mut file_path = include_dir.clone();
            file_path.push(name);
            Ok(ResolvedInclude {
                resolved_name: name.to_owned(),
                content: fs::read_to_string(file_path).map_err(|_| "Couldn't find file.")?,
            })
        });

        let binary_result = self
            .compiler
            .compile_into_spirv(
                &source,
                match options.ty {
                    ShaderType::Vertex => shaderc::ShaderKind::Vertex,
                    ShaderType::Fragment => shaderc::ShaderKind::Fragment,
                    ShaderType::Compute => shaderc::ShaderKind::Compute,
                },
                "shader.glsl",
                "main",
                Some(&additional_options),
            )
            .unwrap();
        Ok(binary_result.as_binary().to_vec())
    }
}

pub struct PipelineCompilerInfo {
    pub compiler: Box<dyn ShaderCompiler>,
    pub include_dir: PathBuf,
    pub debug_name: String,
}

impl Default for PipelineCompilerInfo {
    fn default() -> Self {
        Self {
            compiler: Box::new(ByteToSpirvCompiler),
            include_dir: env::current_dir().unwrap(),
            debug_name: "PipelineCompiler".to_string(),
        }
    }
}

pub struct PipelineCompiler {
    pub(crate) inner: Arc<PipelineCompilerInner>,
}

pub struct PipelineCompilerInner {
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) compiler: Box<dyn ShaderCompiler>,
    pub(crate) include_dir: PathBuf,
    pub(crate) debug_name: String,
}

impl PipelineCompiler {
    pub fn create_graphics_pipeline<'a>(&'a self, info: GraphicsPipelineInfo) -> Result<Pipeline> {
        let PipelineCompilerInner { device, .. } = &*self.inner;

        let DeviceInner { logical_device, .. } = &**device;
        #[cfg(all(feature = "bindless"))]
        let DeviceInner { bindless, .. } = &**device;

        let shader_data = info
            .shaders
            .iter()
            .map(
                |Shader {
                     ty,
                     source,
                     defines,
                 }| {
                    let spv = self
                        .inner
                        .compiler
                        .compile_to_spv(ShaderCompilationOptions {
                            include_dir: &self.inner.include_dir,
                            source,
                            ty: *ty,
                            defines,
                        })?;

                    let shader_module_create_info = {
                        let code_size = 4 * spv.len();

                        let p_code = spv.as_ptr();

                        vk::ShaderModuleCreateInfo {
                            code_size,
                            p_code,
                            ..Default::default()
                        }
                    };

                    unsafe { logical_device.create_shader_module(&shader_module_create_info, None) }
                        .map(|module| (ty, module))
                        .map_err(|_| Error::ShaderCompilationError {
                            message: String::from("Failed to create shader module"),
                        })
                },
            )
            .collect::<Vec<_>>();

        let shader_data = {
            let mut result = vec![];

            for blob in shader_data {
                result.push(blob?);
            }

            result
        };

        let name = ffi::CString::new("main").unwrap();

        let stages = shader_data
            .into_iter()
            .enumerate()
            .map(|(i, (ty, module))| {
                let stage = (*ty).into();

                let p_name = name.as_ptr();

                vk::PipelineShaderStageCreateInfo {
                    stage,
                    module,
                    p_name,
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

        let topology = match info.raster.polygon_mode {
            PolygonMode::Fill => vk::PrimitiveTopology::TRIANGLE_LIST,
            PolygonMode::Line => vk::PrimitiveTopology::LINE_LIST,
            PolygonMode::Point => vk::PrimitiveTopology::POINT_LIST,
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology,
            ..Default::default()
        };

        let rasterization_state = info.raster.into();

        let depth_and_stencil = OptionalDepthStencil(info.depth, info.stencil);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: info.depth.is_some() as _,
            ..depth_and_stencil.into()
        };

        let color_attachments = info
            .color
            .iter()
            .map(|color| vk::PipelineColorBlendAttachmentState {
                blend_enable: color.blend.is_some() as _,
                ..color.blend.unwrap_or_default().into()
            })
            .collect::<Vec<_>>();

        let color_attachment_formats = info
            .color
            .iter()
            .map(|color| color.format.into())
            .collect::<Vec<_>>();

        let color_attachment_count = color_attachments.len() as _;

        let color_blend_state = {
            let attachment_count = color_attachments.len() as u32;

            let p_attachments = color_attachments.as_ptr();

            vk::PipelineColorBlendStateCreateInfo {
                logic_op_enable: false as _,
                attachment_count,
                p_attachments,
                ..Default::default()
            }
        };

        let viewports = [vk::Viewport::default()];

        let scissors = [vk::Rect2D::default()];

        let viewport_state = {
            let viewport_count = 1;

            let p_viewports = viewports.as_ptr();

            let scissor_count = 1;

            let p_scissors = scissors.as_ptr();

            vk::PipelineViewportStateCreateInfo {
                viewport_count,
                p_viewports,
                scissor_count,
                p_scissors,
                ..Default::default()
            }
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = {
            let dynamic_state_count = dynamic_states.len() as u32;

            let p_dynamic_states = dynamic_states.as_ptr();

            vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count,
                p_dynamic_states,
                ..Default::default()
            }
        };

        let descriptor_pool_sizes = match info.binding.clone() {
            BindingState::Binding(bindings) => {
                let mut s = vec![];
                for binding in bindings {
                    s.push(match &binding {
                        Binding::Buffer => vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: MAX_FRAMES_IN_FLIGHT as _,
                        },
                        Binding::Image => vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: MAX_FRAMES_IN_FLIGHT as _,
                        },
                    });
                }
                s
            }
            _ => todo!(),
        };

        let descriptor_pool_create_info = {
            #[cfg(all(feature = "bindless"))]
            let flags = vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
            #[cfg(not(feature = "bindless"))]
            let flags = vk::DescriptorPoolCreateFlags::empty();

            let max_sets = MAX_FRAMES_IN_FLIGHT as _;

            let pool_size_count = descriptor_pool_sizes.len() as u32;

            let p_pool_sizes = descriptor_pool_sizes.as_ptr();

            vk::DescriptorPoolCreateInfo {
                flags,
                max_sets,
                pool_size_count,
                p_pool_sizes,
                ..Default::default()
            }
        };

        let descriptor_pool =
            unsafe { logical_device.create_descriptor_pool(&descriptor_pool_create_info, None) }
                .map_err(|_| Error::CreateDescriptorPool)?;

        let descriptor_set_layout = match info.binding.clone() {
            BindingState::Binding(bindings) => {
                let bindings = bindings
                    .into_iter()
                    .enumerate()
                    .map(|(i, binding)| vk::DescriptorSetLayoutBinding {
                        binding: i as u32,
                        descriptor_type: match binding {
                            Binding::Buffer => vk::DescriptorType::STORAGE_BUFFER,
                            Binding::Image => vk::DescriptorType::STORAGE_IMAGE,
                        },
                        descriptor_count: 1,
                        stage_flags: ShaderStageFlags::ALL,
                        ..Default::default()
                    })
                    .collect::<Vec<_>>();

                let create_info = vk::DescriptorSetLayoutCreateInfo {
                    binding_count: bindings.len() as _,
                    p_bindings: bindings.as_ptr(),
                    ..Default::default()
                };

                unsafe { logical_device.create_descriptor_set_layout(&create_info, None) }
                    .map_err(|_| Error::CreateDescriptorSetLayout)?
            }
            #[cfg(all(feature = "bindless"))]
            BindingState::Bindless => bindless.descriptor_set_layout.clone(),
        };
        let set_layouts = iter::repeat_with(|| descriptor_set_layout.clone())
            .take(MAX_FRAMES_IN_FLIGHT)
            .collect::<Vec<_>>();

        let descriptor_sets = {
            let allocate_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: set_layouts.len() as _,
                p_set_layouts: set_layouts.as_ptr(),

                ..Default::default()
            };
            unsafe { logical_device.allocate_descriptor_sets(&allocate_info) }.map_err(|e| {
                dbg!(e);
                Error::Creation
            })?
        };

        let push_constant = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: info.push_constant_size as _,
        };

        let layout_create_info = {
            let set_layout_count = set_layouts.len() as u32;

            let p_set_layouts = set_layouts.as_ptr();

            let push_constant_range_count = 1;

            let p_push_constant_ranges = &push_constant as *const _;

            vk::PipelineLayoutCreateInfo {
                set_layout_count,
                p_set_layouts,
                push_constant_range_count,
                p_push_constant_ranges,
                ..Default::default()
            }
        };

        let layout = unsafe { logical_device.create_pipeline_layout(&layout_create_info, None) }
            .map_err(|_| Error::Creation)?;

        let depth_attachment_format = info
            .depth
            .map(|x| x.format)
            .unwrap_or(Format::Undefined)
            .into();

        let mut pipeline_rendering_create_info = {
            let p_color_attachment_formats = color_attachment_formats.as_ptr();

            vk::PipelineRenderingCreateInfo {
                color_attachment_count,
                p_color_attachment_formats,
                depth_attachment_format,
                ..Default::default()
            }
        };

        let graphics_pipeline_create_info = {
            let p_next = match &info.binding {
                BindingState::Binding(_) => ptr::null_mut(),
                #[cfg(all(feature = "bindless"))]
                BindingState::Bindless => {
                    &mut pipeline_rendering_create_info as *mut _ as *mut _;
                }
            };

            let stage_count = stages.len() as u32;

            let p_stages = stages.as_ptr();

            let p_vertex_input_state = &vertex_input_state;

            let p_input_assembly_state = &input_assembly_state;

            let p_rasterization_state = &rasterization_state;

            let p_depth_stencil_state = &depth_stencil_state;

            let p_color_blend_state = &color_blend_state;

            let p_viewport_state = &viewport_state;

            let p_dynamic_state = &dynamic_state;

            let render_pass = info
                .render_pass
                .clone()
                .map(|x| x.render_pass)
                .unwrap_or(vk::RenderPass::null());

            vk::GraphicsPipelineCreateInfo {
                p_next,
                stage_count,
                p_stages,
                p_vertex_input_state,
                p_input_assembly_state,
                p_rasterization_state,
                p_depth_stencil_state,
                p_color_blend_state,
                p_viewport_state,
                p_dynamic_state,
                render_pass,
                layout,
                ..Default::default()
            }
        };

        let pipeline = unsafe {
            logical_device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphics_pipeline_create_info],
                None,
            )
        }
        .map_err(|_| Error::Creation)?[0];

        let spec = Spec::Graphics(info);

        let modify = Mutex::new(PipelineModify {
            pipeline,
            layout,
            descriptor_set_layout,
            descriptor_sets,
            descriptor_pool,
        });

        Ok(Pipeline {
            inner: Arc::new(PipelineInner {
                compiler: self.inner.clone(),
                modify,
                spec,
                bind_point: PipelineBindPoint::Graphics,
            }),
        })
    }

    pub fn create_compute_pipeline<'a>(&'a self, info: ComputePipelineInfo) -> Result<Pipeline> {
        let PipelineCompilerInner { device, .. } = &*self.inner;

        let DeviceInner { logical_device, .. } = &**device;
        #[cfg(all(feature = "bindless"))]
        let DeviceInner { bindless, .. } = &**device;

        let spec = Spec::Compute(info.clone());

        let module = {
            let Shader {
                ty,
                source,
                defines,
            } = info.shader;

            let spv = self
                .inner
                .compiler
                .compile_to_spv(ShaderCompilationOptions {
                    include_dir: &self.inner.include_dir,
                    source: &source,
                    ty,
                    defines: &defines,
                })?;

            let shader_module_create_info = {
                let code_size = 4 * spv.len();

                let p_code = spv.as_ptr();

                vk::ShaderModuleCreateInfo {
                    code_size,
                    p_code,
                    ..Default::default()
                }
            };

            unsafe { logical_device.create_shader_module(&shader_module_create_info, None) }
                .map_err(|_| Error::ShaderCompilationError {
                    message: String::from("Failed to create shader module"),
                })?
        };

        let name = ffi::CString::new("main").unwrap();

        let stage = {
            let Shader { ty, .. } = info.shader;

            let stage = ty.into();

            let p_name = name.as_ptr();

            vk::PipelineShaderStageCreateInfo {
                stage,
                module,
                p_name,
                ..Default::default()
            }
        };
        let descriptor_set_layout = match info.binding.clone() {
            BindingState::Binding(bindings) => {
                let bindings = bindings
                    .into_iter()
                    .enumerate()
                    .map(|(i, binding)| vk::DescriptorSetLayoutBinding {
                        binding: i as u32,
                        descriptor_type: match binding {
                            Binding::Buffer => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                            Binding::Image => vk::DescriptorType::STORAGE_IMAGE,
                        },
                        descriptor_count: MAX_FRAMES_IN_FLIGHT as _,
                        stage_flags: ShaderStageFlags::ALL,
                        ..Default::default()
                    })
                    .collect::<Vec<_>>();

                let create_info = vk::DescriptorSetLayoutCreateInfo {
                    binding_count: bindings.len() as _,
                    p_bindings: bindings.as_ptr(),
                    ..Default::default()
                };

                unsafe { logical_device.create_descriptor_set_layout(&create_info, None) }
                    .map_err(|_| Error::CreateDescriptorSetLayout)?
            }
            #[cfg(all(feature = "bindless"))]
            BindingState::Bindless => bindless.descriptor_set_layout.clone(),
        };
        let set_layouts = iter::repeat_with(|| descriptor_set_layout.clone())
            .take(MAX_FRAMES_IN_FLIGHT)
            .collect::<Vec<_>>();

        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 200,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 400,
            },
        ];

        let descriptor_pool_create_info = {
            #[cfg(all(feature = "bindless"))]
            let flags = vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
            #[cfg(not(feature = "bindless"))]
            let flags = vk::DescriptorPoolCreateFlags::empty();

            let max_sets = 200;

            let pool_size_count = descriptor_pool_sizes.len() as u32;

            let p_pool_sizes = descriptor_pool_sizes.as_ptr();

            vk::DescriptorPoolCreateInfo {
                flags,
                max_sets,
                pool_size_count,
                p_pool_sizes,
                ..Default::default()
            }
        };

        let descriptor_pool =
            unsafe { logical_device.create_descriptor_pool(&descriptor_pool_create_info, None) }
                .map_err(|_| Error::CreateDescriptorPool)?;

        let descriptor_sets = {
            let allocate_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: MAX_FRAMES_IN_FLIGHT as u32,
                p_set_layouts: set_layouts.as_ptr(),
                ..Default::default()
            };
            unsafe { logical_device.allocate_descriptor_sets(&allocate_info) }
                .map_err(|_| Error::Creation)?
        };

        let push_constant = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: info.push_constant_size as _,
        };

        let layout_create_info = {
            let set_layout_count = set_layouts.len() as u32;

            let p_set_layouts = set_layouts.as_ptr();

            let push_constant_range_count = 1;

            let p_push_constant_ranges = &push_constant as *const _;

            vk::PipelineLayoutCreateInfo {
                set_layout_count,
                p_set_layouts,
                push_constant_range_count,
                p_push_constant_ranges,
                ..Default::default()
            }
        };

        let layout = unsafe { logical_device.create_pipeline_layout(&layout_create_info, None) }
            .map_err(|_| Error::Creation)?;

        let compute_pipeline_create_info = {
            vk::ComputePipelineCreateInfo {
                stage,
                layout,
                ..Default::default()
            }
        };

        let pipeline = unsafe {
            logical_device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[compute_pipeline_create_info],
                None,
            )
        }
        .map_err(|_| Error::Creation)?[0];

        let modify = Mutex::new(PipelineModify {
            pipeline,
            layout,
            descriptor_set_layout,
            descriptor_sets,
            descriptor_pool,
        });

        Ok(Pipeline {
            inner: Arc::new(PipelineInner {
                compiler: self.inner.clone(),
                modify,
                spec,
                bind_point: PipelineBindPoint::Compute,
            }),
        })
    }

    pub fn refresh_graphics_pipeline<'a>(
        &'a self,
        pipeline: &'a Pipeline,
        shaders: Vec<Shader>,
    ) -> Result<()> {
        let Spec::Graphics(mut info) = pipeline.inner.spec.clone() else {
            Err(Error::InvalidResource)?
        };

        info.shaders = shaders;

        let new_pipeline = self.create_graphics_pipeline(info).unwrap();

        let mut pipeline_modify = pipeline.inner.modify.lock().unwrap();

        let new_pipeline_modify = new_pipeline.inner.modify.lock().unwrap();

        *pipeline_modify = new_pipeline_modify.clone();

        Ok(())
    }

    pub fn refresh_compute_pipeline<'a>(
        &'a self,
        pipeline: &'a Pipeline,
        shader: Shader,
    ) -> Result<()> {
        let Spec::Compute(mut info) = pipeline.inner.spec.clone() else {
            Err(Error::InvalidResource)?
        };

        info.shader = shader;

        let new_pipeline = self.create_compute_pipeline(info).unwrap();

        let mut pipeline_modify = pipeline.inner.modify.lock().unwrap();

        let new_pipeline_modify = new_pipeline.inner.modify.lock().unwrap();

        *pipeline_modify = new_pipeline_modify.clone();

        Ok(())
    }
}

#[derive(Default, Clone, Copy)]
pub enum FrontFace {
    Clockwise,
    #[default]
    CounterClockwise,
}

impl From<FrontFace> for vk::FrontFace {
    fn from(front_face: FrontFace) -> Self {
        match front_face {
            FrontFace::Clockwise => Self::CLOCKWISE,
            FrontFace::CounterClockwise => Self::COUNTER_CLOCKWISE,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line,
    Point,
}

impl From<PolygonMode> for vk::PolygonMode {
    fn from(mode: PolygonMode) -> Self {
        match mode {
            PolygonMode::Fill => Self::FILL,
            PolygonMode::Line => Self::LINE,
            PolygonMode::Point => Self::POINT,
        }
    }
}

bitflags! {
    #[derive(Default, Copy, Clone, Hash, PartialEq, Eq, Debug)]
    pub struct FaceCull : u32 {
        const FRONT = 0x00000002;
        const BACK = 0x00000004;
        const FRONT_AND_BACK = Self::FRONT.bits() | Self::BACK.bits();
    }
}

impl From<FaceCull> for vk::CullModeFlags {
    fn from(cull: FaceCull) -> Self {
        let mut result = vk::CullModeFlags::empty();

        if cull.contains(FaceCull::FRONT) {
            result |= vk::CullModeFlags::FRONT;
        }

        if cull.contains(FaceCull::BACK) {
            result |= vk::CullModeFlags::BACK;
        }

        result
    }
}

bitflags! {
    #[derive(Clone, Copy)]
    pub struct ColorComponent : u32 {
        const R = 0x00000002;
        const G = 0x00000004;
        const B = 0x00000008;
        const A = 0x00000020;
        const ALL = Self::R.bits()
                                | Self::G.bits()
                                | Self::B.bits()
                                | Self::A.bits();
    }
}
impl From<ColorComponent> for vk::ColorComponentFlags {
    fn from(components: ColorComponent) -> Self {
        let mut result = vk::ColorComponentFlags::empty();

        if components.contains(ColorComponent::R) {
            result |= vk::ColorComponentFlags::R;
        }

        if components.contains(ColorComponent::G) {
            result |= vk::ColorComponentFlags::G;
        }

        if components.contains(ColorComponent::B) {
            result |= vk::ColorComponentFlags::B;
        }

        if components.contains(ColorComponent::A) {
            result |= vk::ColorComponentFlags::A;
        }

        result
    }
}

#[derive(Clone, Copy)]
pub struct Raster {
    pub polygon_mode: PolygonMode,
    pub face_cull: FaceCull,
    pub front_face: FrontFace,
    pub depth_clamp: bool,
    pub depth_bias: bool,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
}

impl Default for Raster {
    fn default() -> Self {
        Self {
            polygon_mode: Default::default(),
            face_cull: Default::default(),
            front_face: Default::default(),
            depth_clamp: Default::default(),
            depth_bias: Default::default(),
            depth_bias_constant_factor: Default::default(),
            depth_bias_clamp: Default::default(),
            depth_bias_slope_factor: Default::default(),
            line_width: 1.0,
        }
    }
}

impl From<Raster> for vk::PipelineRasterizationStateCreateInfo {
    fn from(raster: Raster) -> Self {
        Self {
            depth_clamp_enable: raster.depth_clamp as _,
            rasterizer_discard_enable: false as _,
            polygon_mode: raster.polygon_mode.into(),
            cull_mode: raster.face_cull.into(),
            front_face: raster.front_face.into(),
            depth_bias_enable: raster.depth_bias as _,
            depth_bias_constant_factor: raster.depth_bias_constant_factor,
            depth_bias_clamp: raster.depth_bias_clamp,
            depth_bias_slope_factor: raster.depth_bias_slope_factor,
            line_width: raster.line_width,
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SrcAlphaSaturate,
}

impl From<BlendFactor> for vk::BlendFactor {
    fn from(factor: BlendFactor) -> Self {
        match factor {
            BlendFactor::Zero => Self::ZERO,
            BlendFactor::One => Self::ONE,
            BlendFactor::SrcColor => Self::SRC_COLOR,
            BlendFactor::OneMinusSrcColor => Self::ONE_MINUS_SRC_COLOR,
            BlendFactor::DstColor => Self::DST_COLOR,
            BlendFactor::OneMinusDstColor => Self::ONE_MINUS_DST_COLOR,
            BlendFactor::SrcAlpha => Self::SRC_ALPHA,
            BlendFactor::OneMinusSrcAlpha => Self::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DstAlpha => Self::DST_ALPHA,
            BlendFactor::OneMinusDstAlpha => Self::ONE_MINUS_DST_ALPHA,
            BlendFactor::ConstantColor => Self::CONSTANT_COLOR,
            BlendFactor::OneMinusConstantColor => Self::ONE_MINUS_CONSTANT_COLOR,
            BlendFactor::ConstantAlpha => Self::CONSTANT_ALPHA,
            BlendFactor::OneMinusConstantAlpha => Self::ONE_MINUS_CONSTANT_ALPHA,
            BlendFactor::SrcAlphaSaturate => Self::SRC_ALPHA_SATURATE,
        }
    }
}

#[derive(Clone, Copy)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl From<BlendOp> for vk::BlendOp {
    fn from(op: BlendOp) -> Self {
        match op {
            BlendOp::Add => Self::ADD,
            BlendOp::Subtract => Self::SUBTRACT,
            BlendOp::ReverseSubtract => Self::REVERSE_SUBTRACT,
            BlendOp::Min => Self::MIN,
            BlendOp::Max => Self::MAX,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Blend {
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_blend: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_blend: BlendOp,
    pub color_write: ColorComponent,
}

impl Default for Blend {
    fn default() -> Self {
        Self {
            src_color: BlendFactor::One,
            dst_color: BlendFactor::Zero,
            color_blend: BlendOp::Add,
            src_alpha: BlendFactor::One,
            dst_alpha: BlendFactor::Zero,
            alpha_blend: BlendOp::Add,
            color_write: ColorComponent::ALL,
        }
    }
}

impl From<Blend> for vk::PipelineColorBlendAttachmentState {
    fn from(blend: Blend) -> Self {
        Self {
            blend_enable: false as _,
            src_color_blend_factor: blend.src_color.into(),
            dst_color_blend_factor: blend.dst_color.into(),
            color_blend_op: blend.color_blend.into(),
            src_alpha_blend_factor: blend.src_alpha.into(),
            dst_alpha_blend_factor: blend.dst_alpha.into(),
            alpha_blend_op: blend.alpha_blend.into(),
            color_write_mask: blend.color_write.into(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Color {
    pub format: Format,
    pub blend: Option<Blend>,
}

impl Default for Color {
    fn default() -> Self {
        Self {
            format: Default::default(),
            blend: Some(Default::default()),
        }
    }
}

#[derive(Default, Clone, Copy)]
pub enum CompareOp {
    Never,
    #[default]
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

impl From<CompareOp> for vk::CompareOp {
    fn from(op: CompareOp) -> Self {
        match op {
            CompareOp::Never => Self::NEVER,
            CompareOp::Less => Self::LESS,
            CompareOp::Equal => Self::EQUAL,
            CompareOp::LessOrEqual => Self::LESS_OR_EQUAL,
            CompareOp::Greater => Self::GREATER,
            CompareOp::NotEqual => Self::NOT_EQUAL,
            CompareOp::GreaterOrEqual => Self::GREATER_OR_EQUAL,
            CompareOp::Always => Self::ALWAYS,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Depth {
    pub write: bool,
    pub compare: CompareOp,
    pub format: Format,
    pub bounds: (f32, f32),
}

impl Default for Depth {
    fn default() -> Self {
        Self {
            write: true,
            compare: Default::default(),
            format: Format::D32Sfloat,
            bounds: (0.0, 1.0),
        }
    }
}

#[derive(Clone, Copy, Default)]
pub enum StencilOp {
    #[default]
    Keep,
    Zero,
    Replace,
    IncrementAndClamp,
    DecrementAndClamp,
    Invert,
    IncrementAndWrap,
    DecrementAndWrap,
}

impl From<StencilOp> for vk::StencilOp {
    fn from(op: StencilOp) -> Self {
        match op {
            StencilOp::Keep => vk::StencilOp::KEEP,
            StencilOp::Zero => vk::StencilOp::ZERO,
            StencilOp::Replace => vk::StencilOp::REPLACE,
            StencilOp::IncrementAndClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
            StencilOp::DecrementAndClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
            StencilOp::Invert => vk::StencilOp::INVERT,
            StencilOp::IncrementAndWrap => vk::StencilOp::INCREMENT_AND_WRAP,
            StencilOp::DecrementAndWrap => vk::StencilOp::DECREMENT_AND_WRAP,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Stencil {
    pub front: StencilState,
    pub back: StencilState,
}

#[derive(Clone, Copy, Default)]
pub struct StencilState {
    pub fail_op: StencilOp,
    pub pass_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub compare_op: CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

impl From<StencilState> for vk::StencilOpState {
    fn from(ss: StencilState) -> Self {
        Self {
            fail_op: ss.fail_op.into(),
            pass_op: ss.pass_op.into(),
            depth_fail_op: ss.depth_fail_op.into(),
            compare_op: ss.compare_op.into(),
            compare_mask: ss.compare_mask,
            write_mask: ss.write_mask,
            reference: ss.reference,
        }
    }
}

struct OptionalDepthStencil(Option<Depth>, Option<Stencil>);

impl From<OptionalDepthStencil> for vk::PipelineDepthStencilStateCreateInfo {
    fn from(depth_and_stencil: OptionalDepthStencil) -> Self {
        let OptionalDepthStencil(depth, stencil) = depth_and_stencil;
        Self {
            depth_test_enable: depth.is_some() as _,
            depth_write_enable: depth.map(|d| d.write).unwrap_or_default() as _,
            depth_compare_op: depth.map(|d| d.compare).unwrap_or_default().into(),
            min_depth_bounds: depth.map(|d| d.bounds.0).unwrap_or_default() as _,
            max_depth_bounds: depth.map(|d| d.bounds.1).unwrap_or(1.0) as _,
            stencil_test_enable: stencil.is_some() as _,
            front: stencil.map(|s| s.front).unwrap_or_default().into(),
            back: stencil.map(|s| s.back).unwrap_or_default().into(),
            ..Default::default()
        }
    }
}

#[derive(Clone)]
pub enum Binding {
    Buffer,
    Image,
}

#[derive(Clone)]
pub enum BindingState {
    #[cfg(all(feature = "bindless"))]
    Bindless,
    Binding(Vec<Binding>),
}

#[derive(Clone)]
pub struct GraphicsPipelineInfo {
    pub shaders: Vec<Shader>,
    pub color: Vec<Color>,
    pub depth: Option<Depth>,
    pub stencil: Option<Stencil>,
    pub raster: Raster,
    pub push_constant_size: usize,
    pub render_pass: Option<RenderPass>,
    pub binding: BindingState,
    pub debug_name: String,
}

impl Default for GraphicsPipelineInfo {
    fn default() -> Self {
        Self {
            shaders: vec![Default::default()],
            color: vec![],
            depth: None,
            stencil: None,
            raster: Default::default(),
            push_constant_size: 128,
            render_pass: None,
            binding: BindingState::Binding(vec![]),
            debug_name: String::from("Pipeline"),
        }
    }
}

#[derive(Clone)]
pub struct ComputePipelineInfo {
    pub shader: Shader,
    pub push_constant_size: usize,
    pub binding: BindingState,
    pub debug_name: String,
}

impl Default for ComputePipelineInfo {
    fn default() -> Self {
        Self {
            shader: Shader {
                ty: ShaderType::Compute,
                ..Default::default()
            },
            push_constant_size: 128,
            binding: BindingState::Binding(vec![]),
            debug_name: String::from("Pipeline"),
        }
    }
}

bitflags! {
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    pub struct PipelineStage: u32 {
        const TOP_OF_PIPE = 0x00000001;
        const VERTEX_SHADER = 0x00000002;
        const FRAGMENT_SHADER = 0x00000004;
        const EARLY_FRAGMENT_TESTS = 0x00000008;
        const LATE_FRAGMENT_TESTS = 0x00000010;
        const COLOR_ATTACHMENT_OUTPUT = 0x00000020;
        const COMPUTE_SHADER = 0x00000040;
        const TRANSFER = 0x00000080;
        const BOTTOM_OF_PIPE = 0x00000100;
        const HOST = 0x00000200;
        const ALL_GRAPHICS = 0x00000400;
        const ALL_COMMANDS = 0x00000800;
    }
}

impl From<PipelineStage> for vk::PipelineStageFlags {
    fn from(stage: PipelineStage) -> Self {
        let mut result = vk::PipelineStageFlags::empty();

        if stage == PipelineStage::empty() {
            result |= vk::PipelineStageFlags::NONE;
        }

        if stage.contains(PipelineStage::TOP_OF_PIPE) {
            result |= vk::PipelineStageFlags::TOP_OF_PIPE;
        }

        if stage.contains(PipelineStage::VERTEX_SHADER) {
            result |= vk::PipelineStageFlags::VERTEX_SHADER;
        }

        if stage.contains(PipelineStage::FRAGMENT_SHADER) {
            result |= vk::PipelineStageFlags::FRAGMENT_SHADER;
        }

        if stage.contains(PipelineStage::EARLY_FRAGMENT_TESTS) {
            result |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;
        }

        if stage.contains(PipelineStage::LATE_FRAGMENT_TESTS) {
            result |= vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
        }

        if stage.contains(PipelineStage::COLOR_ATTACHMENT_OUTPUT) {
            result |= vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        }

        if stage.contains(PipelineStage::COMPUTE_SHADER) {
            result |= vk::PipelineStageFlags::COMPUTE_SHADER;
        }

        if stage.contains(PipelineStage::TRANSFER) {
            result |= vk::PipelineStageFlags::TRANSFER;
        }

        if stage.contains(PipelineStage::BOTTOM_OF_PIPE) {
            result |= vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        }

        if stage.contains(PipelineStage::HOST) {
            result |= vk::PipelineStageFlags::HOST;
        }

        if stage.contains(PipelineStage::ALL_GRAPHICS) {
            result |= vk::PipelineStageFlags::ALL_GRAPHICS;
        }

        if stage.contains(PipelineStage::ALL_COMMANDS) {
            result |= vk::PipelineStageFlags::ALL_COMMANDS;
        }

        result
    }
}

#[derive(Clone, Copy)]
pub enum PipelineBindPoint {
    Graphics,
    Compute,
}

impl From<PipelineBindPoint> for vk::PipelineBindPoint {
    fn from(bind_point: PipelineBindPoint) -> Self {
        match bind_point {
            PipelineBindPoint::Graphics => vk::PipelineBindPoint::GRAPHICS,
            PipelineBindPoint::Compute => vk::PipelineBindPoint::COMPUTE,
        }
    }
}

#[derive(Clone)]
pub(crate) enum Spec {
    Graphics(GraphicsPipelineInfo),
    Compute(ComputePipelineInfo),
}

#[derive(Clone)]
pub struct Pipeline {
    pub(crate) inner: Arc<PipelineInner>,
}

pub struct PipelineInner {
    pub(crate) compiler: Arc<PipelineCompilerInner>,
    pub(crate) modify: Mutex<PipelineModify>,
    pub(crate) bind_point: PipelineBindPoint,
    pub(crate) spec: Spec,
}

#[derive(Clone)]
pub(crate) struct PipelineModify {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) descriptor_sets: Vec<vk::DescriptorSet>,
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) descriptor_pool: vk::DescriptorPool,
}
