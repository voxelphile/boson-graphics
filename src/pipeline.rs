use crate::device::DeviceInner;
use crate::prelude::*;

use std::borrow;
use std::default::default;
use std::env;
use std::ffi;
use std::fmt;
use std::fs;
use std::path;
use std::process;
use std::sync::{Arc, Mutex};

use ash::vk;

use bitflags::bitflags;

use lazy_static::lazy_static;

pub type Spv = Vec<u32>;

///Requires a name, a type, and the includes
#[derive(Clone, Copy)]
pub struct Shader<'a>(pub ShaderType, pub &'a str, pub &'a [&'a str]);

impl Default for Shader<'_> {
    fn default() -> Self {
        Self(ShaderType::Vertex, "", &[])
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
    input_path: &'a path::Path,
    output_path: &'a path::Path,
    ty: ShaderType,
    defines: &'a [&'a str],
}

pub enum ShaderCompiler {
    None,
    Glslc { language: ShaderLanguage },
    Dxc { language: ShaderLanguage },
}

impl Default for ShaderCompiler {
    fn default() -> Self {
        Self::None
    }
}

impl ShaderCompiler {
    pub fn glslc(info: ShaderCompilerInfo) -> Self {
        let ShaderCompilerInfo { language } = info;

        ShaderCompiler::Glslc { language }
    }

    pub fn dxc(info: ShaderCompilerInfo) -> Self {
        let ShaderCompilerInfo { language } = info;

        ShaderCompiler::Dxc { language }
    }

    pub(crate) fn compile_to_spv(&self, options: ShaderCompilationOptions) -> Result<Spv> {
        let vulkan_path = env::var("VULKAN_SDK").map_err(|e| Error::ShaderCompilerNotFound)?;

        match self {
            ShaderCompiler::None => Ok(fs::read(options.output_path)
                .expect("failed to read shader compilation output")
                .chunks(4)
                .map(|a| u32::from_le_bytes(a.try_into().unwrap()))
                .collect::<Vec<_>>()),
            ShaderCompiler::Glslc { language } => {
                let glslc_path = path::PathBuf::from(vulkan_path);

                let source_code = fs::read_to_string(options.input_path).map_err(|_| {
                    Error::ShaderCompilationError {
                        message: format!("failed to read shader: {}", options.input_path.display()),
                    }
                })?;

                let mut temporary_path = options.input_path.to_path_buf();

                temporary_path.pop();
                temporary_path.push(&format!(
                    "{}",
                    options.input_path.file_stem().unwrap().to_string_lossy()
                ));

                fs::remove_file(temporary_path.clone());

                let mut modified_code = source_code.clone();

                modified_code =
                    modified_code.replacen("\n", &format!("\n#define {}\r\n", options.ty), 1);

                for define in options.defines {
                    modified_code =
                        modified_code.replacen("\n", &format!("\n#define {}\r\n", define), 1);
                }

                fs::write(temporary_path.clone(), modified_code);

                let glslc = process::Command::new("glslc")
                    .current_dir(glslc_path)
                    .arg("-g")
                    .arg(format!("-fshader-stage={}", options.ty))
                    .arg("--target-env=vulkan1.3")
                    .arg("-O")
                    .arg("-c")
                    .arg(&temporary_path)
                    .arg("-o")
                    .arg(&options.output_path)
                    .spawn()
                    .map_err(|e| {
                        dbg!(e);
                        Error::ShaderCompilationError {
                            message: String::from("failed to spawn glslc"),
                        }
                    })?;

                let glslc =
                    glslc
                        .wait_with_output()
                        .map_err(|_| Error::ShaderCompilationError {
                            message: String::from("failed to wait on glslc"),
                        })?;

                let spv = glslc
                    .status
                    .exit_ok()
                    .map(|_| {
                        fs::read(options.output_path)
                            .expect("failed to read shader compilation output")
                            .chunks(4)
                            .map(|a| u32::from_le_bytes(a.try_into().unwrap()))
                            .collect::<Vec<_>>()
                    })
                    .map_err(|e| {
                        dbg!(e);
                        Error::ShaderCompilationError {
                            message: String::from("glsl error"),
                        }
                    })?;

                fs::remove_file(temporary_path);

                Ok(spv)
            }
            _ => todo!(),
        }
    }

    pub(crate) fn language(&self) -> Option<ShaderLanguage> {
        Some(match self {
            Self::None => None?,
            Self::Glslc { language } => *language,
            Self::Dxc { language } => *language,
        })
    }

    pub(crate) fn extension(&self) -> Option<&'static str> {
        Some(match self.language()? {
            ShaderLanguage::Glsl => "glsl",
            ShaderLanguage::Hlsl => "hlsl",
        })
    }
}

lazy_static! {
    static ref CURRENT_PATH: path::PathBuf =
        env::current_dir().expect("failed to get current directory");
}

pub struct PipelineCompilerInfo<'a> {
    pub compiler: ShaderCompiler,
    pub source_path: &'a path::Path,
    pub asset_path: &'a path::Path,
    pub debug_name: &'a str,
}

impl Default for PipelineCompilerInfo<'_> {
    fn default() -> Self {
        Self {
            compiler: Default::default(),
            source_path: &CURRENT_PATH,
            asset_path: &CURRENT_PATH,
            debug_name: "PipelineCompiler",
        }
    }
}

pub struct PipelineCompiler {
    pub(crate) inner: Arc<PipelineCompilerInner>,
}

pub struct PipelineCompilerInner {
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) compiler: ShaderCompiler,
    pub(crate) source_path: path::PathBuf,
    pub(crate) asset_path: path::PathBuf,
    pub(crate) debug_name: String,
}

impl PipelineCompiler {
    pub fn create_graphics_pipeline<'a, const S: usize, const C: usize>(
        &'a self,
        info: GraphicsPipelineInfo<'a, S, C>,
    ) -> Result<Pipeline<'a, S, C>> {
        let PipelineCompilerInner { device, .. } = &*self.inner;

        let DeviceInner {
            logical_device,
            descriptor_set_layout,
            ..
        } = &**device;

        let shader_data = info
            .shaders
            .iter()
            .map(|Shader(ty, name, defines)| {
                let extension = self.inner.compiler.extension().unwrap_or("");

                let mut input_path = self.inner.source_path.clone();

                input_path.push(name);
                input_path.set_extension(extension);

                let mut output_path = self.inner.asset_path.clone();

                let short_ty = format!("{}", ty).chars().take(4).collect::<String>();

                output_path.push(name);
                output_path.set_extension(format!("{}.spv", &short_ty));

                let spv = self
                    .inner
                    .compiler
                    .compile_to_spv(ShaderCompilationOptions {
                        input_path: &input_path,
                        output_path: &output_path,
                        ty: *ty,
                        defines: *defines,
                    })?;

                let shader_module_create_info = {
                    let code_size = 4 * spv.len();

                    let p_code = spv.as_ptr();

                    vk::ShaderModuleCreateInfo {
                        code_size,
                        p_code,
                        ..default()
                    }
                };

                unsafe { logical_device.create_shader_module(&shader_module_create_info, None) }
                    .map(|module| (ty, module))
                    .map_err(|_| Error::ShaderCompilationError {
                        message: String::from("Failed to create shader module"),
                    })
            })
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
                    ..default()
                }
            })
            .collect::<Vec<_>>();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..default()
        };

        let rasterization_state = info.raster.into();

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: info.depth.is_some() as _,
            ..info.depth.unwrap_or_default().into()
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
                ..default()
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
                ..default()
            }
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = {
            let dynamic_state_count = dynamic_states.len() as u32;

            let p_dynamic_states = dynamic_states.as_ptr();

            vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count,
                p_dynamic_states,
                ..default()
            }
        };

        let set_layouts = [*descriptor_set_layout];

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
                ..default()
            }
        };

        let layout = unsafe { logical_device.create_pipeline_layout(&layout_create_info, None) }
            .map_err(|_| Error::Creation)?;

        let depth_attachment_format = vk::Format::D32_SFLOAT;

        let mut pipeline_rendering_create_info = {
            let p_color_attachment_formats = color_attachment_formats.as_ptr();

            vk::PipelineRenderingCreateInfo {
                color_attachment_count,
                p_color_attachment_formats,
                depth_attachment_format,
                ..default()
            }
        };

        let graphics_pipeline_create_info = {
            let p_next = &mut pipeline_rendering_create_info as *mut _ as *mut _;

            let stage_count = stages.len() as u32;

            let p_stages = stages.as_ptr();

            let p_vertex_input_state = &vertex_input_state;

            let p_input_assembly_state = &input_assembly_state;

            let p_rasterization_state = &rasterization_state;

            let p_depth_stencil_state = &depth_stencil_state;

            let p_color_blend_state = &color_blend_state;

            let p_viewport_state = &viewport_state;

            let p_dynamic_state = &dynamic_state;

            let render_pass = vk::RenderPass::null();

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
                ..default()
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

        let modify = Mutex::new(PipelineModify { pipeline, layout });

        Ok(Pipeline {
            inner: Arc::new(PipelineInner {
                compiler: self.inner.clone(),
                modify,
                spec,
                bind_point: PipelineBindPoint::Graphics,
            }),
        })
    }

    pub fn create_compute_pipeline<'a>(
        &'a self,
        info: ComputePipelineInfo<'a>,
    ) -> Result<Pipeline<1, 0>> {
        let PipelineCompilerInner { device, .. } = &*self.inner;

        let DeviceInner {
            logical_device,
            descriptor_set_layout,
            ..
        } = &**device;

        let module = {
            let Shader(ty, name, defines) = info.shader;

            let extension = self.inner.compiler.extension().unwrap_or("");

            let mut input_path = self.inner.source_path.clone();

            input_path.push(name);
            input_path.set_extension(extension);

            let mut output_path = self.inner.asset_path.clone();

            let short_ty = format!("{}", ty).chars().take(4).collect::<String>();

            output_path.push(name);
            output_path.set_extension(format!("{}.spv", &short_ty));

            let spv = self
                .inner
                .compiler
                .compile_to_spv(ShaderCompilationOptions {
                    input_path: &input_path,
                    output_path: &output_path,
                    ty,
                    defines,
                })?;

            let shader_module_create_info = {
                let code_size = 4 * spv.len();

                let p_code = spv.as_ptr();

                vk::ShaderModuleCreateInfo {
                    code_size,
                    p_code,
                    ..default()
                }
            };

            unsafe { logical_device.create_shader_module(&shader_module_create_info, None) }
                .map_err(|_| Error::ShaderCompilationError {
                    message: String::from("Failed to create shader module"),
                })?
        };

        let name = ffi::CString::new("main").unwrap();

        let stage = {
            let Shader(ty, _, _) = info.shader;

            let stage = ty.into();

            let p_name = name.as_ptr();

            vk::PipelineShaderStageCreateInfo {
                stage,
                module,
                p_name,
                ..default()
            }
        };

        let set_layouts = [*descriptor_set_layout];

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
                ..default()
            }
        };

        let layout = unsafe { logical_device.create_pipeline_layout(&layout_create_info, None) }
            .map_err(|_| Error::Creation)?;

        let compute_pipeline_create_info = {
            vk::ComputePipelineCreateInfo {
                stage,
                layout,
                ..default()
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

        let spec = Spec::Compute(info);

        let modify = Mutex::new(PipelineModify { pipeline, layout });

        Ok(Pipeline {
            inner: Arc::new(PipelineInner {
                compiler: self.inner.clone(),
                modify,
                spec,
                bind_point: PipelineBindPoint::Compute,
            }),
        })
    }

    pub fn refresh_graphics_pipeline<'a, const S: usize, const C: usize>(
        &'a self,
        pipeline: &'a Pipeline<'a, S, C>,
    ) -> Result<()> {
        let Spec::Graphics(info) = pipeline.inner.spec else {
            Err(Error::InvalidResource)?
        };

        let new_pipeline = self.create_graphics_pipeline(info).unwrap();

        let mut pipeline_modify = pipeline.inner.modify.lock().unwrap();

        let new_pipeline_modify = new_pipeline.inner.modify.lock().unwrap();

        *pipeline_modify = *new_pipeline_modify;

        Ok(())
    }

    pub fn refresh_compute_pipeline<'a>(&'a self, pipeline: &'a Pipeline<'a, 1, 0>) -> Result<()> {
        let Spec::Compute(info) = pipeline.inner.spec else {
            Err(Error::InvalidResource)?
        };

        let new_pipeline = self.create_compute_pipeline(info).unwrap();

        let mut pipeline_modify = pipeline.inner.modify.lock().unwrap();

        let new_pipeline_modify = new_pipeline.inner.modify.lock().unwrap();

        *pipeline_modify = *new_pipeline_modify;

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
    #[derive(Default)]
    pub struct FaceCull : u32 {
        const FRONT = 0x00000002;
        const BACK = 0x00000004;
        const FRONT_AND_BACK = Self::FRONT.bits | Self::BACK.bits;
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
    pub struct ColorComponent : u32 {
        const R = 0x00000002;
        const G = 0x00000004;
        const B = 0x00000008;
        const A = 0x00000020;
        const ALL = Self::R.bits
                                | Self::G.bits
                                | Self::B.bits
                                | Self::A.bits;
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
            polygon_mode: default(),
            face_cull: default(),
            front_face: default(),
            depth_clamp: default(),
            depth_bias: default(),
            depth_bias_constant_factor: default(),
            depth_bias_clamp: default(),
            depth_bias_slope_factor: default(),
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
            ..default()
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
            format: default(),
            blend: Some(default()),
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
    pub bounds: (f32, f32),
}

impl Default for Depth {
    fn default() -> Self {
        Self {
            write: true,
            compare: default(),
            bounds: (0.0, 1.0),
        }
    }
}

impl From<Depth> for vk::PipelineDepthStencilStateCreateInfo {
    fn from(depth: Depth) -> Self {
        Self {
            depth_test_enable: true as _,
            depth_write_enable: depth.write as _,
            depth_compare_op: depth.compare.into(),
            min_depth_bounds: depth.bounds.0 as _,
            max_depth_bounds: depth.bounds.1 as _,
            ..default()
        }
    }
}

#[derive(Clone, Copy)]
pub struct GraphicsPipelineInfo<'a, const S: usize, const C: usize> {
    pub shaders: [Shader<'a>; S],
    pub color: [Color; C],
    pub depth: Option<Depth>,
    pub raster: Raster,
    pub push_constant_size: usize,
    pub debug_name: &'a str,
}

impl<const S: usize, const C: usize> Default for GraphicsPipelineInfo<'_, S, C> {
    fn default() -> Self {
        Self {
            shaders: [default(); S],
            color: [default(); C],
            depth: None,
            raster: default(),
            push_constant_size: 128,
            debug_name: "Pipeline",
        }
    }
}

#[derive(Clone, Copy)]
pub struct ComputePipelineInfo<'a> {
    pub shader: Shader<'a>,
    pub push_constant_size: usize,
    pub debug_name: &'a str,
}

impl Default for ComputePipelineInfo<'_> {
    fn default() -> Self {
        Self {
            shader: Shader(ShaderType::Compute, "default", &[]),
            push_constant_size: 128,
            debug_name: "Pipeline",
        }
    }
}

bitflags! {
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

#[derive(Clone, Copy)]
pub(crate) enum Spec<'a, const S: usize, const C: usize> {
    Graphics(GraphicsPipelineInfo<'a, S, C>),
    Compute(ComputePipelineInfo<'a>),
}

pub struct Pipeline<'a, const S: usize, const C: usize> {
    pub(crate) inner: Arc<PipelineInner<'a, S, C>>,
}

pub struct PipelineInner<'a, const S: usize, const C: usize> {
    pub(crate) compiler: Arc<PipelineCompilerInner>,
    pub(crate) modify: Mutex<PipelineModify>,
    pub(crate) bind_point: PipelineBindPoint,
    pub(crate) spec: Spec<'a, S, C>,
}

#[derive(Clone, Copy)]
pub(crate) struct PipelineModify {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
}
