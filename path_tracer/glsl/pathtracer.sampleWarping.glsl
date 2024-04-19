
vec3 squareToDiskConcentric(vec2 xi) {
    float phi, r;
    float a = 2.0 * xi.x - 1.0;
    float b = 2.0 * xi.y - 1.0;

    if (a == 0 && b == 0) {
        return vec3(0, 0, 0);
    }

    if (abs(a) > abs(b)) {
        r = a;
        phi = (PI / 4.0) * (b / a);
    } else {
        r = b;
        phi = (PI / 2.0) - (PI / 4.0) * (a / b);
    }


    return vec3(r * cos(phi), r * sin(phi), 0.0);
}

vec3 squareToHemisphereCosine(vec2 xi) {
    vec3 diskSample = squareToDiskConcentric(xi);
    float z = sqrt(1.0 - diskSample.x * diskSample.x - diskSample.y * diskSample.y);
    return vec3(diskSample.x, diskSample.y, z);
}

float squareToHemisphereCosinePDF(vec3 sample) {

    return sample.z / PI;
}

vec3 squareToSphereUniform(vec2 sample) {
    float z = 1.0 - 2.0 * sample.x;
    float r = sqrt( 1.0 - z * z);
    float phi = 2.0 * PI * sample.y;
    return vec3(r * cos(phi), r * sin(phi), z);
}

float squareToSphereUniformPDF(vec3 sample) {
    return 1.0 / (4.0 * PI);
}
