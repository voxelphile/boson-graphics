#version 450
//This does all the shader magic for connecting to boson
#include "../../boson.glsl"

struct Time {
  f32 time_since_startup;
};

decl_buffer(Time)

struct Push {
  Buffer(Time) time_buffer;
};

decl_push_constant(Push)

#if shader_type == shader_type_vertex

layout(location = 0) out f32vec4 color;


void main() {
    f32 time = deref(push.time_buffer).time_since_startup;

    f32 x = f32(i32(1) - i32(gl_VertexIndex)) * 0.5;
    f32 y = f32((i32(gl_VertexIndex) & i32(1)) * 2 - 1) * 0.5;

    if (gl_VertexIndex == 0) {
        color = f32vec4(1.0, 0.0, 0.0, 1.0);
    } else if (gl_VertexIndex == 1) {
        color = f32vec4(0.0, 1.0, 0.0, 1.0);
    } else {
        color = f32vec4(0.0, 0.0, 1.0, 1.0);
    }

    f32vec2 _2d_position = f32vec2(x, y);

    f32 range = 0.2;
    f32 oscillate = mod(time / 20.0, range);
    if(oscillate < range / 2.0) {
        oscillate = range - oscillate;
    }

    _2d_position.x += oscillate - range / 2.0;

    gl_Position = f32vec4(_2d_position, 0.0, 1.0);
}

#elif shader_type == shader_type_fragment

layout(location = 0) in vec4 color;

layout(location = 0) out vec4 result;

void main() {
    result = color;
}

#endif