//#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable

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
    const float theta = 2.f * M_PI_F * u2;

    vec3 w = normal;
    vec3 axis = fabs(w.x) > 0.1f ? (vec3)(0.0f, 1.0f, 0.0f) : (vec3)(1.0f, 0.0f, 0.0f);
    vec3 u = normalize(cross(axis, w));
    vec3 v = cross(w, u);

    /* use the coordinte frame and random numbers to compute the next ray direction */
    return normalize(u * cos(theta)*r + v*sin(theta)*r + w*sqrt(1.0f - u1));
}

#define NUM_BOUNCES 4
#define NUM_SAMPLES 4

typedef struct spath_progress
{
    unsigned int sample_num;
    unsigned int bounce_num;
    vec3 mask;
    vec3 accum_color;
} __attribute__((aligned (16))) spath_progress; //NOTE: space for two more 32 bit dudes

__kernel void segmented_path_trace_init(
    __global vec4* out_tex,
    __global ray* ray_buffer,
    __global ray* ray_origin_buffer,
    __global kd_tree_collision_result* kd_results,
    __global kd_tree_collision_result* kd_source_results,
    __global spath_progress* spath_data,

    const __global material* material_buffer,

//Mesh
    const __global mesh* meshes,
    image1d_buffer_t indices,
    image1d_buffer_t vertices,
    image1d_buffer_t normals,

    const unsigned int width,
    const unsigned int random_value)
{
    const vec4 sky = (vec4) (0.16, 0.2, 0.2, 0)*2;
    int x = get_global_id(0)%width;
    int y = get_global_id(0)/width;
    int offset = (x+y*width);

    kd_tree_collision_result res = kd_results[offset];
    ray r = ray_buffer[offset];
    ray_origin_buffer[offset] = r;
    kd_source_results[offset] = res;

    spath_progress spd;
    spd.mask = (vec3)(1.0f, 1.0f, 1.0f);
    spd.accum_color = (vec3) (0, 0, 0);


    if(res.t==0)
    {
        out_tex[offset] += sky;
    }

    unsigned int seed1 = random_value * x;
    unsigned int seed2 = random_value * y;

#pragma unroll   //NOTE: NVIDIA plugin
    for(int i = 0; i < 7; i++)
        get_random(&seed1, &seed2);


    float rand1 = get_random(&seed1, &seed2);
    float rand2 = get_random(&seed1, &seed2);


    int4 i1 = read_imagei(indices, (int)res.triangle_index);
    int4 i2 = read_imagei(indices, (int)res.triangle_index+1);
    int4 i3 = read_imagei(indices, (int)res.triangle_index+2);

    mesh m = meshes[i1.w];

    material mat = material_buffer[m.material_index];

    vec3 pos = r.orig + r.dir*res.t;

    vec3 normal =
        read_imagef(normals, (int)i1.y).xyz*(1-res.u-res.v)+
        read_imagef(normals, (int)i2.y).xyz*res.u+
        read_imagef(normals, (int)i3.y).xyz*res.v;

    spd.mask *= mat.colour;

    ray sr;
    vec3 sample_dir = cosineSampleHemisphere(rand1, rand2, normal);

    //sweet spot for epsilon
    sr.orig = pos + normal * 0.0001f;
    sr.dir = sample_dir;

    ray_buffer[offset] = sr;
    spath_data[offset] = spd;
}

__kernel void segmented_path_trace(
    __global vec4* out_tex,
    __global ray* ray_buffer,
    __global ray* ray_origin_buffer,
    __global kd_tree_collision_result* kd_results,
    __global kd_tree_collision_result* kd_source_results,
    __global spath_progress* spath_data,

    const __global unsigned int* random_buffer,

    const __global material* material_buffer,

    //Mesh
    const __global mesh* meshes,
    image1d_buffer_t indices,
    image1d_buffer_t vertices,
    image1d_buffer_t normals,

    const unsigned int width,
    const unsigned int random_value)
{
    const vec4 sky = (vec4) (0.16, 0.2, 0.2, 0);

    int x = get_global_id(0)%width;
    int y = get_global_id(0)/width;
    int offset = (x+y*width);

    spath_progress spd = spath_data[offset];

    //get this from the cpu, this is very hacky
    if(spd.sample_num==2048)
    {
        ray nr;
        nr.orig = (vec3)(0);
        nr.dir  = (vec3)(0);
        ray_buffer[offset] = nr;
        return;
    }
    kd_tree_collision_result res;
    ray r;

    res = kd_results[offset];
    r = ray_buffer[offset];

    //RETRIEVE DATA
    int4 i1 = read_imagei(indices, (int)res.triangle_index);
    int4 i2 = read_imagei(indices, (int)res.triangle_index+1);
    int4 i3 = read_imagei(indices, (int)res.triangle_index+2);
    mesh m = meshes[i1.w];
    material mat = material_buffer[m.material_index];
    vec3 pos = r.orig + r.dir*res.t;
    //pos = (vec3) (0, 0, -2);

    vec3 normal =
        read_imagef(normals, (int)i1.y).xyz*(1-res.u-res.v)+
        read_imagef(normals, (int)i2.y).xyz*res.u+
        read_imagef(normals, (int)i3.y).xyz*res.v;


    unsigned int seed1 = random_buffer[offset]*random_value;
    unsigned int seed2 = random_buffer[offset];

     //MESSY CODE!
    float rand1 = get_random(&seed1, &seed2);
    float rand2 = get_random(&seed2, &seed1);

    ray sr;

    vec3 sample_dir = cosineSampleHemisphere(rand1, rand2, normal);
    sr.orig = pos + normal * 0.0001f; //sweet spot for epsilon
    sr.dir = sample_dir;

    //THE NEXT PART
    if(res.t==0)
    {
        spd.bounce_num = NUM_BOUNCES;
        spd.accum_color += spd.mask * sky.xyz;
    }
    else
    {
        //NOTE: janky emission, if reflectivity is 1 emission is 2 (only for tests)
        spd.accum_color += spd.mask * (float)(mat.reflectivity==1.f)*2.f; //NOTE: ADD EMMISION

        spd.mask *= mat.colour;

        spd.mask *= dot(sr.dir, normal);
    }

    spd.bounce_num++;

    if(spd.bounce_num >= NUM_BOUNCES)
    {

        spd.bounce_num = 0;
        spd.sample_num++;
#ifdef _WIN32
        out_tex[offset] += (vec4)(spd.accum_color, 1);
#else
        out_tex[offset] += (vec4)(spd.accum_color.zyx, 1);
#endif
        //START OF NEW


        res = kd_source_results[offset];
        r = ray_origin_buffer[offset];
        spd.mask = (vec3)(1.0f, 1.0f, 1.0f);
        spd.accum_color = (vec3) (0, 0, 0);


        if(res.t==0)
        {
            out_tex[offset] += sky;
        }

        i1 = read_imagei(indices, (int)res.triangle_index);
        i2 = read_imagei(indices, (int)res.triangle_index+1);
        i3 = read_imagei(indices, (int)res.triangle_index+2);
        m = meshes[i1.w];
        mat = material_buffer[m.material_index];
        pos = r.orig + r.dir*res.t;

        normal =
            read_imagef(normals, (int)i1.y).xyz*(1-res.u-res.v)+
            read_imagef(normals, (int)i2.y).xyz*res.u+
            read_imagef(normals, (int)i3.y).xyz*res.v;

        spd.mask *= mat.colour;

        //TODO: just add an emmision value in material
        if( (float)(mat.reflectivity==1.))
        {
            spd.accum_color += spd.mask*2;
        }

        sample_dir = cosineSampleHemisphere(rand1, rand2, normal);
        sr.orig = pos + normal * 0.0001f; //sweet spot for epsilon
        sr.dir = sample_dir;
    }

    ray_buffer[offset] = sr;

    spath_data[offset] = spd;

}

__kernel void path_trace(
    __global vec4* out_tex,
    const __global ray* ray_buffer,
    const __global material* material_buffer,
    const __global sphere* spheres,
    const __global plane* planes,
    //Mesh
    const __global mesh* meshes,
    image1d_buffer_t indices,
    image1d_buffer_t vertices,
    image1d_buffer_t normals,

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

    int x = get_global_id(0);
    int y = get_global_id(1);

    int offset = (x+y*width);

    ray r;
    r = ray_buffer[offset];
    r.orig = pos.xyz;
    union {
		float f;
		unsigned int ui;
	} res;

    //fill up the mantissa.
    res.f = (float)magic*M_PI_F+x;
    unsigned int seed1 = res.ui + (int)(sin((float)x)*7.1f);

    res.f = (float)magic*M_PI_F+y;
    unsigned int seed2 = y + (int)(sin((float)res.ui)*7*3.f);

    collision_result initial_result;
    if(!collide_all(r, &initial_result, s, MESH_SCENE_DATA))
    {
        out_tex[x+y*width] = sky;
        return;
    }

    vec3 fin_colour = (vec3)(0.0f, 0.0f, 0.0f);
    for(int i = 0; i < NUM_SAMPLES; i++)
    {

        vec3 accum_color = (vec3)(0.0f, 0.0f, 0.0f);
        vec3 mask        = (vec3)(1.0f, 1.0f, 1.0f);
        ray sr;
        float rand1 = get_random(&seed1, &seed2);
        float rand2 = get_random(&seed1, &seed2);


        vec3 sample_dir =  cosineSampleHemisphere(rand1, rand2, initial_result.normal);

        //sweet spot for epsilon
        sr.orig = initial_result.point + initial_result.normal * 0.0001f;

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
    const unsigned int sample)
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

    out_tex[offset] = (uchar4) ((unsigned char)x2,
                                (unsigned char)0,
                                (unsigned char)0,
                                (unsigned char)1.f);

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


    //Using -0.5 so the colours average around 0, better precision.

    vec4 old = (out_tex[offset] - 0.5f) * ((float)sample - 1.f);
    vec4 new = fresh_frame_tex[offset] - 0.5f;

    out_tex[offset] = (old+new) / (float) sample;

}

__kernel void xorshift_batch(__global unsigned int* data)
{
    //get_global_id is just a register, not a function
    uint d = data[get_global_id(0)];
    data[get_global_id(0)] = ((d << 1) | (d >> (sizeof(int)*8 - 1)))+1;//circular shift +1
}

__kernel void f_buffer_to_byte_buffer_avg(
    __global unsigned int* out_tex,
    __global vec4* fresh_frame_tex,
    __global spath_progress* spath_data,
    const unsigned int width,
    const unsigned int sample_num)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width);

    vec4 data   = fresh_frame_tex[offset];
    vec4 colour = data.w==0 ? (vec4)(0,0,0,0) : data.xyzw/data.w;

    out_tex[offset] = get_colour(colour);
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
