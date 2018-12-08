
vec3 uniformSampleHemisphere(const float r1, const float r2)
{
    // cos(theta) = r1 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
    float sinTheta = sqrt(1 - r1 * r1);
    float phi = 2 * M_PI * r2;
    float x = sinTheta * cos(phi);
    float z = sinTheta * sin(phi);
    return (vec3)(x, r1, z);
}
vec3 cosineSampleHemisphere(float u1, float u2, vec3 normal)
{
    const float r = sqrt(u1);
    const float theta = 2 * M_PI * u2;

    vec3 w = normal;
    vec3 axis = fabs(w.x) > 0.1f ? (vec3)(0.0f, 1.0f, 0.0f) : (vec3)(1.0f, 0.0f, 0.0f);
    vec3 u = normalize(cross(axis, w));
    vec3 v = cross(w, u);

    /* use the coordinte frame and random numbers to compute the next ray direction */
    return normalize(u * cos(theta)*r + v*sin(theta)*r + w*sqrt(1.0f - u1));

}

#define NUM_BOUNCES 4
#define NUM_SAMPLES 64
__kernel void path_trace(
    __global vec4* out_tex,
    const __global float* ray_buffer,
    const __global material* material_buffer,
    const __global sphere* spheres,
    const __global plane* planes,
//Mesh
    const __global mesh* meshes,
    const __global int* indices,
    const __global vec3* vertices,
    const __global vec3* normals,
    /* const __global vec2* texcoords, */
    /* , */
    const unsigned int width,
    const unsigned int height,
    const vec4 pos,
    unsigned int magic)
{
    const vec4 sky = (vec4) (0.16, 0.2, 0.2, 0);
    //return;
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;
    int ray_offset = offset*3;

    ray r;
    r.orig = pos.xyz;
    r.dir.x = ray_buffer[ray_offset]; //NOTE: unoptomized memory access.
    r.dir.y = ray_buffer[ray_offset+1];
    r.dir.z = ray_buffer[ray_offset+2];



    union {
		float f;
		unsigned int ui;
	} res;

    res.f = (float)magic*M_PI+x;//get some decimals.
    unsigned int seed1 = res.ui + (int)(sin((float)x)*7.f);

    res.f = (float)magic*M_PI+y;//get some decimals.
    unsigned int seed2 = res.ui + (int)(sin((float)y)*7.f);

    //unsigned int seed1 = (int)((magic+1)%x)*7+magic+x/*sin((float)magic))*/,
    //    seed2 = (int)((magic+1)%y)*7+magic+y/*cos((float)magic))*/;
    vec3 fin_colour = (vec3)(0.0f, 0.0f, 0.0f);
    for(int i = 0; i < NUM_SAMPLES; i++)
    {
        vec3 accum_color = (vec3)(0.0f, 0.0f, 0.0f);
        vec3 mask        = (vec3)(1.0f, 1.0f, 1.0f);
        ray sr = r;

        for(int bounces = 0; bounces < NUM_BOUNCES; bounces++)
        {
            collision_result result;
            if(!collide_all(sr, &result, material_buffer, spheres, planes, meshes, indices,
                            vertices, normals))
            {
                accum_color += mask * sky.xyz;//(vec3) (1.f,1.f,1.f);//(vec3)(0.15f, 0.15f, 0.25f);
                break;
            }


            float rand1 = get_random(&seed1, &seed2);
            float rand2 = get_random(&seed1, &seed2); //seed but even more seedy :)
            // float3 normal_facing = dot(result.normal, sr.dir) < 0.0f ? result.normal : result.normal * (-1.0f);


            vec3 sample_dir =  cosineSampleHemisphere(rand1, rand2, result.normal);

            sr.orig = result.point + result.normal * 0.0001f; //sweet spot for epsilon
            sr.dir = sample_dir;

            accum_color += mask * (float)(result.mat.reflectivity==1.)*2; //NOTE: EMMISION


            mask *= result.mat.colour;

            mask *= dot(sample_dir, result.normal);

            //vec4 colour = shade(result);

        }

        fin_colour += clamp(accum_color,0.f,1.f) * (1.f/NUM_SAMPLES);
    }

    //   colour = mix(colour, colours[0],  result.mat.reflectivity);

    out_tex[x+y*width] = (vec4) (fin_colour, 0);//get_colour((vec4) (fin_colour, 0) );
    //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.

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

    //t_tex[offset] = (char4) (mi( (float4)fresh_frame_tex[offset], (float4)out_tex[offset], (float)sample/255))

    float4 temp = mix((float4)(
                          (float)fresh_frame_tex[offset].x,
                          (float)fresh_frame_tex[offset].y,
                          (float)fresh_frame_tex[offset].z,
                          (float)fresh_frame_tex[offset].w),
                      (float4)(
                          (float)out_tex[offset].x,
                          (float)out_tex[offset].y,
                          (float)out_tex[offset].z,
                          (float)out_tex[offset].w), (float)sample/24.f);
    /*vec4 temp =  (float)(
        (float)fresh_frame_tex[offset].x,
        (float)fresh_frame_tex[offset].y,
        (float)fresh_frame_tex[offset].z,
        (float)fresh_frame_tex[offset].w)/12.f;*/
    out_tex[offset] = (uchar4) ((unsigned char)temp.x,
                                (unsigned char)temp.y,
                                (unsigned char)temp.z,
                                (unsigned char)temp.w);
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
    out_tex[offset] = mix(fresh_frame_tex[offset], out_tex[offset], ((float)sample+1.f)/(float)num_samples);
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
    //if(fresh_frame_tex[offset].x!=fresh_frame_tex[offset].y)printf("n (%f,%f,%f)\n", fresh_frame_tex[offset].x,fresh_frame_tex[offset].y,fresh_frame_tex[offset].z);
    out_tex[offset] = get_colour(fresh_frame_tex[offset]);
}
