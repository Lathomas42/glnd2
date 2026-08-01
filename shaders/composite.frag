#version 430 core

#define MAX_CHANNELS 8

in vec2 vTex;
out vec4 fragColor;

uniform sampler2D uTex[MAX_CHANNELS];
uniform int uCount;
uniform int uEnabled[MAX_CHANNELS];
uniform float uBlack[MAX_CHANNELS];
uniform float uWhite[MAX_CHANNELS];
uniform float uGamma[MAX_CHANNELS];
uniform vec3 uColor[MAX_CHANNELS];

void main() {
    vec3 acc = vec3(0.0);
    for (int i = 0; i < MAX_CHANNELS; i++) {
        if (i >= uCount) break;
        if (uEnabled[i] == 0) continue;

        float v = texture(uTex[i], vTex).r;
        float black = uBlack[i];
        float white = max(uWhite[i], black + 1e-6);
        float g = max(uGamma[i], 0.01);

        float x = clamp((v - black) / (white - black), 0.0, 1.0);
        x = pow(x, 1.0 / g);
        acc += x * uColor[i];
    }
    fragColor = vec4(clamp(acc, 0.0, 1.0), 1.0);
}
