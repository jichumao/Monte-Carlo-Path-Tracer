
const float FOVY = 19.5f * PI / 180.0;


Ray rayCast() {
    vec2 offset = vec2(rng(), rng());
    vec2 ndc = (vec2(gl_FragCoord.xy) + offset) / vec2(u_ScreenDims);
    ndc = ndc * 2.f - vec2(1.f);

    float aspect = u_ScreenDims.x / u_ScreenDims.y;
    vec3 ref = u_Eye + u_Forward;
    vec3 V = u_Up * tan(FOVY * 0.5);
    vec3 H = u_Right * tan(FOVY * 0.5) * aspect;
    vec3 p = ref + H * ndc.x + V * ndc.y;

    return Ray(u_Eye, normalize(p - u_Eye));
}

//vec3 Li_Naive0(Ray ray) {

//    vec3 L = vec3(0.0);

//    Intersection isect = sceneIntersect(ray);
//    if (isect.t < INFINITY && isect.material.type == DIFFUSE_REFL) {
//        vec3 wiW;
//        float pdf;
//        int sampledType;

//        vec3 f = Sample_f_diffuse(isect.material.albedo, vec2(rng(), rng()), isect.nor, wiW, pdf, sampledType);

//        L = 0.5 * (normalize(wiW) + vec3(1.0));
//    }

//    return L;
//}

//vec3 Li_Naive(Ray ray) {
//    vec3 L = vec3(0.0); // Accumulated light
//    vec3 throughput = vec3(1.0); // Throughput for the ray at each bounce
//    Intersection isect;

//    for(int depth = 0; depth < MAX_DEPTH; ++depth) {
//        isect = sceneIntersect(ray);
//        // Check for no intersection and possibly return environment light
//        if (isect.t == INFINITY) {
//            break;
//        }

//        // Direct light contribution
//        if (isect.Le != vec3(0.0)) {
//            L += throughput * isect.Le;
//            break; // Light source was hit, no further bounces needed
//        }

//        // Sample the BRDF
//        vec3 wiW;
//        float pdf;
//        int sampledType;

//        vec3 bsdf = Sample_f(isect, -ray.direction, vec2(rng(), rng()), wiW, pdf, sampledType);

//        if (pdf > 0.0) {
//            throughput *= bsdf * AbsDot(wiW, isect.nor) / pdf;
//        } else {
//            break; // Terminate the path as it's no longer contributing
//        }

//        // Spawn the next ray
//        ray = SpawnRay(ray.origin + isect.t * ray.direction, wiW);
//    }

//    return L;
//}

//vec3 Li_Direct_Simple(Ray ray) {
//    vec3 L = vec3(0.0); // Accumulated light
//    Intersection isect = sceneIntersect(ray);

//    // Check for no intersection
//    if (isect.t == INFINITY) {
//        return L; // Environment light could be added here if needed
//    }

//    // Check for intersection with a light source
//    if (isect.Le != vec3(0.0)) {
//        return isect.Le; // Light source was hit directly
//    }

//    // Direct light contribution using importance sampling
//    vec3 wiW;
//    float pdf;
//    int chosenLightIdx;
//    int chosenLightID;
//    vec3 Li = Sample_Li(ray.origin + isect.t * ray.direction, isect.nor, wiW, pdf, chosenLightIdx, chosenLightID); // Sample light source
//    //vec3 bsdf = Sample_f(isect, -ray.direction, vec2(rng(), rng()), wiW, pdf, sampledType);

//    vec3 brdf = f(isect, -ray.direction, wiW);

//    if (pdf > 0.0 && Li != vec3(0.0)) {
//        // Assuming no occlusion between the point and the light source
//        L += Li * brdf * AbsDot(wiW, isect.nor) / pdf;
//    }

//    return L;
//}

//vec3 Li_DirectMIS(Ray ray) {
//    vec3 L = vec3(0.0);
//    Intersection isect = sceneIntersect(ray);

//    if (isect.t == INFINITY) return L; // No intersection
//    if (isect.Le != vec3(0.0)) {
//        return isect.Le; // Light source was hit directly
//    }

//    vec3 view_point;
//    int sampledType;
//    // Light Source Sampling
//    vec3 wiW_lightSampled;
//    vec3 bsdf_lightSampled;

//    float pdf_bsdf_lightSampled;
//    float pdf_light_lightSampled;

//    int chosenLightIdx, chosenLightID;

//    view_point = ray.origin + isect.t * ray.direction;
//    // Get 2 diff pdf
//    // Create a ray, get Li & pdf_bsdf_lightSampled
//    vec3 Li_lightSampled = Sample_Li(view_point,
//                                     isect.nor, wiW_lightSampled,
//                                     pdf_light_lightSampled,
//                                     chosenLightIdx,
//                                     chosenLightID);
//    // Get BSDF term
//    bsdf_lightSampled = f(isect, -ray.direction, wiW_lightSampled);

//    // Compute the light-sampled color
//    if (pdf_light_lightSampled > 0.0 && Li_lightSampled != vec3(0.0)) {

//        pdf_bsdf_lightSampled = Pdf(isect, -ray.direction, wiW_lightSampled);
//        // Assuming no occlusion between the point and the light source
//        float weight_light = PowerHeuristic(1, pdf_light_lightSampled,
//                                            1, pdf_bsdf_lightSampled);

//        L +=  weight_light * Li_lightSampled * bsdf_lightSampled * AbsDot(wiW_lightSampled, isect.nor) / pdf_light_lightSampled;
//    }


//    // BRDF Sampling (similar process as above, inverted for sampling strategy)
//    vec3 wiW_bsdfSampled;
//    vec3 bsdf_bsdfSampled;
//    float pdf_bsdf_bsdfSampled;

//    bsdf_bsdfSampled = Sample_f(isect,
//                                     -ray.direction,
//                                     vec2(rng(), rng()),
//                                     wiW_bsdfSampled,
//                                     pdf_bsdf_bsdfSampled,
//                                     sampledType);

//    if (pdf_bsdf_bsdfSampled > 0.0 && bsdf_bsdfSampled != vec3(0.)) {

//        // Compute the ShadowRay
//        AreaLight light = areaLights[chosenLightIdx];
//        Ray newray = SpawnRay(view_point,
//                           wiW_bsdfSampled);
//        Intersection nextIsect = sceneIntersect(newray);

//        // If wj intersects the chosen light source at all
//        if (nextIsect.obj_ID == light.ID) {

//            vec3 Li_bsdfSampled = areaLights[chosenLightIdx].Le ;

//            float pdf_light_bsdfSampled = Pdf_Li(view_point,
//                                                 isect.nor,
//                                                 wiW_bsdfSampled,
//                                                 chosenLightIdx);

//            float weight_light = PowerHeuristic(1, pdf_bsdf_bsdfSampled,
//                                                1, pdf_light_bsdfSampled);

//            L += weight_light * Li_bsdfSampled * bsdf_bsdfSampled * AbsDot(wiW_bsdfSampled, isect.nor) / pdf_bsdf_bsdfSampled;
//        }
//    }

//    return L;
//}

vec3 ComputeDirectLightMIS(Intersection isect,Ray ray) {
    vec3 L = vec3(0.0);

    if (isect.Le != vec3(0.0)) {
        return isect.Le; // Light source was hit directly
    }

    vec3 view_point;
    int sampledType;
    // Light Source Sampling
    vec3 wiW_lightSampled;
    vec3 bsdf_lightSampled;

    float pdf_bsdf_lightSampled;
    float pdf_light_lightSampled;

    int chosenLightIdx, chosenLightID;
    int choosenLightType;
    view_point = ray.origin + isect.t * ray.direction;
    // Get 2 diff pdf
    // Create a ray, get Li & pdf_bsdf_lightSampled

    if (isect.t == INFINITY) {
        return L;
    } // No intersection

    vec3 Li_lightSampled = Sample_Li(view_point,
                                     isect.nor, wiW_lightSampled,
                                     pdf_light_lightSampled,
                                     chosenLightIdx,
                                     chosenLightID,
                                     choosenLightType);
    // Get BSDF term
    bsdf_lightSampled = f(isect, -ray.direction, wiW_lightSampled);

    // Compute the light-sampled color
    if (pdf_light_lightSampled > 0.0 && Li_lightSampled != vec3(0.0)) {

        pdf_bsdf_lightSampled = Pdf(isect, -ray.direction, wiW_lightSampled);
        // Assuming no occlusion between the point and the light source
        float weight_light = PowerHeuristic(1, pdf_light_lightSampled,
                                            1, pdf_bsdf_lightSampled);

        if (choosenLightType == 1){
#if N_AREA_LIGHTS
           L +=  weight_light * Li_lightSampled * bsdf_lightSampled * AbsDot(wiW_lightSampled, isect.nor) / pdf_light_lightSampled;
#endif
        }else if (choosenLightType == 2) {
#if N_POINT_LIGHTS
           L +=  Li_lightSampled * bsdf_lightSampled * AbsDot(wiW_lightSampled, isect.nor) / pdf_light_lightSampled;
#endif
        }else if(choosenLightType == 3){
#if N_SPOT_LIGHTS
           L +=  Li_lightSampled * bsdf_lightSampled * AbsDot(wiW_lightSampled, isect.nor) / pdf_light_lightSampled;
#endif
        }else{
            // environment map
           L +=  Li_lightSampled * bsdf_lightSampled * AbsDot(wiW_lightSampled, isect.nor) / pdf_light_lightSampled;
        }

    }


    // BRDF Sampling (similar process as above, inverted for sampling strategy)
    vec3 wiW_bsdfSampled;
    vec3 bsdf_bsdfSampled;
    float pdf_bsdf_bsdfSampled;

    bsdf_bsdfSampled = Sample_f(isect,
                                     -ray.direction,
                                     vec2(rng(), rng()),
                                     wiW_bsdfSampled,
                                     pdf_bsdf_bsdfSampled,
                                     sampledType);

    if (pdf_bsdf_bsdfSampled > 0.0 && bsdf_bsdfSampled != vec3(0.)) {

        // Compute the ShadowRay
        if (choosenLightType == 1){
#if N_AREA_LIGHTS
            AreaLight light = areaLights[chosenLightIdx];

            Ray newray = SpawnRay(view_point,
                               wiW_bsdfSampled);
            Intersection nextIsect = sceneIntersect(newray);

            // If wj intersects the chosen light source at all
            if (nextIsect.obj_ID == light.ID) {

                vec3 Li_bsdfSampled = light.Le ;

                float pdf_light_bsdfSampled = Pdf_Li(view_point,
                                                     isect.nor,
                                                     wiW_bsdfSampled,
                                                     chosenLightIdx);
                float weight_light = PowerHeuristic(1, pdf_bsdf_bsdfSampled,
                                                    1, pdf_light_bsdfSampled);

                L += weight_light * Li_bsdfSampled * bsdf_bsdfSampled * AbsDot(wiW_bsdfSampled, isect.nor) / pdf_bsdf_bsdfSampled;
            }
#endif
        }else if (choosenLightType == 2) {
#if N_POINT_LIGHTS
            //PointLight light = pointLights[chosenLightIdx - N_AREA_LIGHTS];
#endif
        }else if(choosenLightType == 3){
#if N_SPOT_LIGHTS
            //SpotLight light = spotLights[chosenLightIdx - N_AREA_LIGHTS - N_POINT_LIGHTS];
#endif
        }else{
            // environment map
        }

    }

    return L;
}

vec3 Li_Full(Ray ray) {
    vec3 accum_color = vec3(0.0);
    vec3 throughput = vec3(1.0);
    bool prev_was_specular = false;
    Intersection isect;

    for(int depth = 0; depth < MAX_DEPTH; ++depth) {

        vec3 wo = -ray.direction;

        vec3 direct_Li = vec3(0.);

        if (isect.material.type == SPEC_REFL ||
            isect.material.type == SPEC_TRANS ||
            isect.material.type == SPEC_GLASS) {
            prev_was_specular = true;
        }else{
            // Get the next Intersection
            isect = sceneIntersect(ray);
            direct_Li = ComputeDirectLightMIS(isect, ray);
            prev_was_specular = false;
        }

        isect = sceneIntersect(ray);
        if (isect.t == INFINITY) {
            vec2 uv = sampleSphericalMap(ray.direction);
            vec3 env_Le = texture(u_EnvironmentMap, uv).rgb;
            accum_color += throughput * env_Le;
            break;
            return vec3(0.);
        }

        if (isect.Le != vec3(0.)) {
            if (prev_was_specular || depth == 0) {
                return isect.Le * throughput + accum_color;
            }
            return accum_color;
        }

        // Get the INDIRECT
        vec3 wiW;
        float pdf;
        int sampledType;

        vec3 bsdf = Sample_f(isect, -ray.direction, vec2(rng(), rng()), wiW, pdf, sampledType);


        if (pdf > 0.f && bsdf!= vec3(0.)) {
            float lambert = AbsDot(wiW, isect.nor);
            ray = SpawnRay(ray.origin + isect.t * ray.direction, wiW);
            throughput *= bsdf * lambert / pdf;
            accum_color += direct_Li * throughput;
        }else{
            break;
        }

    }

    return accum_color;
}



void main()
{
    seed = uvec2(u_Iterations, u_Iterations + 1) * uvec2(gl_FragCoord.xy);

    Ray ray = rayCast();

    //vec3 thisIterationColor = Li_Naive(ray);
    //vec3 thisIterationColor = Li_Direct_Simple(ray);
    //vec3 thisIterationColor = Li_DirectMIS(ray);
    vec3 thisIterationColor = Li_Full(ray);

    vec3 prevAccumColor = texelFetch(u_AccumImg, ivec2(gl_FragCoord.xy), 0).rgb;
    // TODO: Set out_Col to the weighted sum of thisIterationColor
    // and all previous iterations' color values.
    // Refer to pathtracer.defines.glsl for what variables you may use
    // to acquire the needed values.

    float iterationFactor = 1.0 / (float(u_Iterations));
    vec3 totalColor = mix(prevAccumColor, thisIterationColor, iterationFactor);

    out_Col = vec4(totalColor, 1.0);
}
