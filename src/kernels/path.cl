
vec3 uniformSampleHemisphere(const float r1, const float r2)
{
    float sinTheta = sqrt(1 - r1 * r1);
    float phi = 2 * M_PI_F * r2;
    float x = sinTheta * cos(phi);
    float z = sinTheta * sin(phi);
    return (vec3)(x, r1, z);
}
vec3 cosineSampleHemisphere(float u1, float u2, vec3 normal)
{
    const float r = sqrt(u1);
    const float theta = 2 * M_PI_F * u2;

    vec3 w = normal;
    vec3 axis = fabs(w.x) > 0.1f ? (vec3)(0.0f, 1.0f, 0.0f) : (vec3)(1.0f, 0.0f, 0.0f);
    vec3 u = normalize(cross(axis, w));
    vec3 v = cross(w, u);

    /* use the coordinte frame and random numbers to compute the next ray direction */
    return normalize(u * cos(theta)*r + v*sin(theta)*r + w*sqrt(1.0f - u1));
}

#define NUM_BOUNCES 2
#define NUM_SAMPLES 4
__kernel void path_trace(
    __global vec4* out_tex,
    const __global float* ray_buffer,
    const __global material* material_buffer,
    const __global sphere* spheres,
    const __global plane* planes,
//Mesh
    const __global mesh* meshes,
    image1d_t indices,
    image1d_t vertices,
    image1d_t normals,
    /* const __global vec2* texcoords, */
    const unsigned int width,
    const vec4 pos,
    unsigned int magic)
{
    scene s;
    s.material_buffer = material_buffer;
    s.spheres         = spheres;
    s.planes          = planes;
    s.meshes          = meshes;


    const vec4 sky = (vec4) (0.16, 0.2, 0.2, 0);
    //return;
    int x = get_global_id(0);
    int y = get_global_id(1);
    //int x  = id%width+ get_global_offset(0)%total_width;
    //int y  = id/width/* + get_global_offset(0)/total_width*/;
    int offset = (x+y*width);
    int ray_offset = offset*3;

    ray r;
    r.orig = pos.xyz;
    r.dir.x = ray_buffer[ray_offset]; //NOTE: unoptimized memory access.
    r.dir.y = ray_buffer[ray_offset+1];
    r.dir.z = ray_buffer[ray_offset+2];



    union {
		float f;
		unsigned int ui;
	} res;

    res.f = (float)magic*M_PI_F+x;//fill up the mantissa.
    unsigned int seed1 = res.ui + (int)(sin((float)x)*7.1f);

    res.f = (float)magic*M_PI_F+y;
    unsigned int seed2 = y + (int)(sin((float)res.ui)*7*3.f);

    collision_result initial_result;
    if(!collide_all(r, &initial_result, s, MESH_SCENE_DATA))
    {
        out_tex[x+y*width] = sky;
        return;
    }
    barrier(0); //good ?

    vec3 fin_colour = (vec3)(0.0f, 0.0f, 0.0f);
    for(int i = 0; i < NUM_SAMPLES; i++)
    {

        vec3 accum_color = (vec3)(0.0f, 0.0f, 0.0f);
        vec3 mask        = (vec3)(1.0f, 1.0f, 1.0f);
        ray sr;
        float rand1 = get_random(&seed1, &seed2);
        float rand2 = get_random(&seed1, &seed2);


        vec3 sample_dir =  cosineSampleHemisphere(rand1, rand2, initial_result.normal);
        sr.orig = initial_result.point + initial_result.normal * 0.0001f; //sweet spot for epsilon
        sr.dir = sample_dir;
        mask *= initial_result.mat.colour;
        for(int bounces = 0; bounces < NUM_BOUNCES; bounces++)
        {
            collision_result result;
            if(!collide_all(sr, &result, s, MESH_SCENE_DATA))
            {
                accum_color += mask * sky.xyz;
                break;
            }


            rand1 = get_random(&seed1, &seed2);
            rand2 = get_random(&seed1, &seed2);

            sample_dir =  cosineSampleHemisphere(rand1, rand2, result.normal);

            sr.orig = result.point + result.normal * 0.0001f; //sweet spot for epsilon
            sr.dir = sample_dir;

            //NOTE: janky emission, if reflectivity is 1 emission is 2 (only for tests)
            accum_color += mask * (float)(result.mat.reflectivity==1.)*2; //NOTE: EMMISION


            mask *= result.mat.colour;

            mask *= dot(sample_dir, result.normal);
        }

        //barrier(0); //good?

        accum_color = clamp(accum_color, 0.f, 1.f);

        fin_colour += accum_color * (1.f/NUM_SAMPLES);
    }
    #ifdef _WIN32
    out_tex[offset] = (vec4)(fin_colour, 1);
    #else
    out_tex[offset] = (vec4)(fin_colour.zyx, 1);
    #endif
}


__kernel void buffer_average(
    __global uchar4* out_tex,
    __global uchar4* fresh_frame_tex,
    const unsigned int width,
    const unsigned int height,
    const unsigned int sample
    /*const unsigned int num_samples*/)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width);
    //        (n - 1) m[n-1] + a[n]
    // m[n] = ---------------------
    //                  n

    float x2 = ((float)sample-1.f)*( (float)out_tex[offset].x + (float)fresh_frame_tex[sample].x)  /
               (float)sample;

//wo
    /*float4 temp = mix((float4)(
                            (float)fresh_frame_tex[offset].x,
                          (float)fresh_frame_tex[offset].y,
                          (float)fresh_frame_tex[offset].z,
                          (float)fresh_frame_tex[offset].w),
                      (float4)(
                          (float)out_tex[offset].x,
                          (float)out_tex[offset].y,
                          (float)out_tex[offset].z,
                          (float)out_tex[offset].w), 0.5f+((float)sample/2048.f/2.f));// );*/
    /*vec4 temp =  (float)(
        (float)fresh_frame_tex[offset].x,
        (float)fresh_frame_tex[offset].y,
        (float)fresh_frame_tex[offset].z,
        (float)fresh_frame_tex[offset].w)/12.f;*/
    out_tex[offset] = (uchar4) ((unsigned char)x2,
                                (unsigned char)0,
                                (unsigned char)0,
                                (unsigned char)1.f);
/*
        fresh_frame_tex[offset]/(unsigned char)(1.f/(1-(float)sample/255))
        + out_tex[offset]/(unsigned char)(1.f/((float)sample/255));*/
}

__kernel void f_buffer_average(
    __global vec4* out_tex,
    __global vec4* fresh_frame_tex,
    const unsigned int width,
    const unsigned int height,
    const unsigned int num_samples,
    const unsigned int sample)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width);

    //        (n - 1) m[n-1] + a[n]
    // m[n] = ---------------------
    //                  n

    out_tex[offset] = ((sample-1) * out_tex[offset] + fresh_frame_tex[offset]) / (float) sample;


    //out_tex[offset] = mix(fresh_frame_tex[offset], out_tex[offset],
    //((float)sample)/(float)num_samples);
}

__kernel void f_buffer_to_byte_buffer(
    __global unsigned int* out_tex,
    __global vec4* fresh_frame_tex,
    const unsigned int width,
    const unsigned int height)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width);

    out_tex[offset] = get_colour(fresh_frame_tex[offset]);
}
