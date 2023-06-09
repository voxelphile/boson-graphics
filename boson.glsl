#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_image_int64 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require

#define EPSILON 1e-2

#define b32 bool
#define i32 int
#define u32 uint
#define f32 float
#define u16 uint16_t
#define i16 int16_t

#define b32vec2 bvec2
#define b32vec3 bvec3
#define b32vec4 bvec4
#define f32 float
#define f32vec2 vec2
#define f32mat2x2 mat2x2

#define f32mat2x3 mat2x3
#define f32mat2x4 mat2x4
#define f32vec3 vec3
#define f32mat3x2 mat3x2
#define f32mat3x3 mat3x3
#define f32mat3x4 mat3x4
#define f32vec4 vec4
#define f32mat4x2 mat4x2
#define f32mat4x3 mat4x3
#define f32mat4x4 mat4x4
#define i32 int
#define u32 uint
#define i64 int64_t
#define u64 uint64_t
#define i32vec2 ivec2
#define u32vec2 uvec2
#define i32vec3 ivec3
#define u32vec3 uvec3
#define i32vec4 ivec4
#define u32vec4 uvec4

#define DEVICE_ADDRESS_BUFFER_BINDING 4
#define SPECIAL_BUFFER_BINDING 3
#define SPECIAL_IMAGE_BINDING 2

struct BufferId {
	u32 buffer_id_value;
};

struct ImageId {
	u32 image_id_value;
};

layout(scalar, binding = DEVICE_ADDRESS_BUFFER_BINDING, set = 0) readonly buffer BufferDeviceAddressBuffer
{
    u64 addresses[];
} buffer_device_address_buffer;

#define _buffer_reference_layout layout(buffer_reference, scalar, buffer_reference_align = 4)
#define _storage_image_layout layout(binding = SPECIAL_IMAGE_BINDING, set = 0)

#define decl_buffer(STRUCT_TYPE) \
    _buffer_reference_layout buffer Buffer##STRUCT_TYPE { \
        STRUCT_TYPE value;  \
    };								

#define _decl_image_kind(name, kind, type) 																\
	_storage_image_layout uniform name ImageTable##kind##type[];														\
	struct Image##kind##type																	\
	{																				\
		ImageId id;																	\
	};														

#define _decl_image_type(kind)																		\
	_decl_image_kind(image##kind, kind, f32)															\
	_decl_image_kind(uimage##kind, kind, u32)															\
	_decl_image_kind(iimage##kind, kind, i32)															\
	_decl_image_kind(uimage##kind, kind, u16)															\
	_decl_image_kind(iimage##kind, kind, i16)

_decl_image_type(1D)
_decl_image_type(2D)
_decl_image_type(3D)

#define decl_push_constant(name) \
    layout(scalar, push_constant) uniform _PUSH_CONSTANT \
    {\
		name push;\
	};

#define Buffer(name) Buffer##name
#define Image(kind, type) Image##kind##type

#define deref(name) name.value

#define _register_image_kind(kind, dim, type)                                                     						\
    type##vec4 imageLoad(Image##kind##type image, i32vec##dim index)             				\
    {                                                                                                                                                              	\
        return imageLoad(ImageTable##kind##type[image.id.image_id_value], index);                                             				\
    }                                                                                                                                                              	\
    void imageStore(Image##kind##type image, i32vec##dim index, type##vec4 data) 				\
    {                                                                                                                                                              	\
        imageStore(ImageTable##kind##type[image.id.image_id_value], index, data);                                             				\
    }                                                                                                                                                              	\
    i32vec##dim imageSize(Image##kind##type image)                                                                             				\
    {                                                                                                                                                             	\
        return imageSize(ImageTable##kind##type[image.id.image_id_value]);                                                                          			\
    }

#define _register_image_kind2(kind, dim, type1, type2)                                                     						\
    type1##vec4 imageLoad(Image##kind##type2 image, i32vec##dim index)             				\
    {                                                                                                                                                              	\
        return imageLoad(ImageTable##kind##type2[image.id.image_id_value], index);                                             				\
    }                                                                                                                                                              	\
    void imageStore(Image##kind##type2 image, i32vec##dim index, type1##vec4 data) 				\
    {                                                                                                                                                              	\
        imageStore(ImageTable##kind##type2[image.id.image_id_value], index, data);                                             				\
    }                                                                                                                                                              	\
    i32vec##dim imageSize(Image##kind##type2 image)                                                                             				\
    {                                                                                                                                                             	\
        return imageSize(ImageTable##kind##type2[image.id.image_id_value]);                                                                          			\
    }

#define _register_image_types(kind, dim)                     \
    _register_image_kind(kind, dim, f32)  \
    _register_image_kind(kind, dim, i32) \
    _register_image_kind(kind, dim, u32) \
    _register_image_kind2(kind, dim, i32, i16) \
    _register_image_kind2(kind, dim, u32, u16)

_register_image_types(2D, 2)
_register_image_types(3D, 3)