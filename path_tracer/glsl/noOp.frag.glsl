#version 330 core

uniform sampler2D u_Texture;
uniform vec2 u_ScreenDims;
uniform int u_Iterations;

in vec3 fs_Pos;
in vec2 fs_UV;

out vec4 out_Col;
void main()
{
//    vec4 color = texture(u_Texture, fs_UV);
//    out_Col = vec4(color.rgb, 1.);

    // TODO: Apply the Reinhard operator and gamma correction
    // before outputting color.

    vec4 color = texture(u_Texture, fs_UV);
    vec3 hdrColor = color.rgb;
    vec3 ldrColor = hdrColor / (hdrColor + vec3(1.0));

    // Apply gamma correction (assuming a gamma of 2.2)
    vec3 gammaCorrectedColor = pow(ldrColor, vec3(1.0/2.2));

    // Output the final color
    out_Col = vec4(gammaCorrectedColor, 1.0);


}
