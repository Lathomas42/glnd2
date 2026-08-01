#version 430 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;

out vec2 vTex;

uniform vec2 uScale;
uniform vec2 uPan;

void main() {
    vec2 pos = aPos * uScale + uPan;
    gl_Position = vec4(pos, 0.0, 1.0);
    vTex = aTex;
}
