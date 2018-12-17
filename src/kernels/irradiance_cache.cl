/******************************************/
/* NOTE: Irradiance Caching is Incomplete */
/******************************************/

/**********************/
/* Irradiance Caching */
/**********************/

__kernel void ic_hemisphere_sample(

    )
{



}

__kernel void ic_screen_textures(
    __write_only image2d_t pos_tex,
    __write_only image2d_t nrm_tex,
    const unsigned int width,
    const unsigned int height,
    const __global float* ray_buffer,
    const vec4 pos,
    const __global material* material_buffer,
    const __global sphere* spheres,
    const __global plane* planes,
    const __global mesh* meshes,
    image1d_t indices,
    image1d_t vertices,
    image1d_t normals)
{
    scene s;
    s.material_buffer = material_buffer;
    s.spheres         = spheres;
    s.planes          = planes;
    s.meshes          = meshes;


    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;
    int ray_offset = offset*3;

    ray r;
    r.orig = pos.xyz; //NOTE: slow unaligned memory access.
    r.dir.x = ray_buffer[ray_offset];
    r.dir.y = ray_buffer[ray_offset+1];
    r.dir.z = ray_buffer[ray_offset+2];

    collision_result result;
    if(!collide_all(r, &result, s, MESH_SCENE_DATA))
    {
        write_imagef(pos_tex, (int2)(x,y), (vec4)(0));
        write_imagef(nrm_tex, (int2)(x,y), (vec4)(0));
        return;
    }

    write_imagef(pos_tex, (int2)(x,y), (vec4)(result.point,0)); //Maybe ???
    write_imagef(nrm_tex, (int2)(x,y), (vec4)(result.normal,0));

    /* pos_tex[offset] = (vec4)(result.point,0); */
    /* nrm_tex[offset] = (vec4)(result.normal,0); */
}



__kernel void generate_discontinuity(
    image2d_t pos_tex,
    image2d_t nrm_tex,
    __global float* out_tex,
    const float k,
    const float intensity,
    const unsigned int width,
    const unsigned int height)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;

    //NOTE: this is fine for edges because the sampler is clamped

    //Positions
    vec4 pm = read_imagef(pos_tex, sampler, (int2)(x,y));
    vec4 pu = read_imagef(pos_tex, sampler, (int2)(x,y+1));
    vec4 pd = read_imagef(pos_tex, sampler, (int2)(x,y-1));
    vec4 pr = read_imagef(pos_tex, sampler, (int2)(x+1,y));
    vec4 pl = read_imagef(pos_tex, sampler, (int2)(x-1,y));

    //NOTE: slow doing this many distance calculations
    float posDiff = max(distance(pu,pm),
                        max(distance(pd,pm),
                            max(distance(pr,pm),
                                distance(pl,pm))));
    posDiff = clamp(posDiff, 0.f, 1.f);
    posDiff *= intensity;

    //Normals
    vec4 nm = read_imagef(nrm_tex, sampler, (int2)(x,y));

    vec4 nu = read_imagef(nrm_tex, sampler, (int2)(x,y+1));
    vec4 nd = read_imagef(nrm_tex, sampler, (int2)(x,y-1));
    vec4 nr = read_imagef(nrm_tex, sampler, (int2)(x+1,y));
    vec4 nl = read_imagef(nrm_tex, sampler, (int2)(x-1,y));
    //NOTE: slow doing this many distance calculations
    float nrmDiff = max(distance(nu,nm),
                        max(distance(nd,nm),
                            max(distance(nr,nm),
                                distance(nl,nm))));
    nrmDiff = clamp(nrmDiff, 0.f, 1.f);
    nrmDiff *= intensity;

    out_tex[offset] = k*nrmDiff+posDiff;
}

__kernel void float_average(
    __global float* in_tex,
    __global float* out_tex,
    const unsigned int width,
    const unsigned int height,
    const int total)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;

    out_tex[offset] += in_tex[offset]/(float)total;

}


__kernel void mip_single_upsample( //nearest neighbour upsample.
    __global float* in_tex,
    __global float* out_tex,
    const unsigned int width, //Of upsampled
    const unsigned int height)//Of upsampled
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;

    out_tex[offset] = in_tex[(x+y*width)/2]; //truncated
}

__kernel void mip_upsample( //nearest neighbour upsample.
    image2d_t in_tex,
    __write_only image2d_t out_tex, //NOTE: not having __write_only caused it to crash without err
    const unsigned int width, //Of upsampled
    const unsigned int height)//Of upsampled
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;

    write_imagef(out_tex, (int2)(x,y),
                 read_imagef(in_tex, sampler, (float2)((float)x/2.f, (float)y/2.f)));
}

__kernel void mip_upsample_scaled( //nearest neighbour upsample.
    image2d_t in_tex,
    __write_only image2d_t out_tex,
    const int s,
    const unsigned int width, //Of upsampled
    const unsigned int height)//Of upsampled
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    float factor = pow(2.f, (float)s);
    write_imagef(out_tex, (int2)(x,y),
                 read_imagef(in_tex, sampler, (float2)((float)x/factor, (float)y/factor)));
}
__kernel void mip_single_upsample_scaled( //nearest neighbour upsample.
    __global float* in_tex,
    __global float* out_tex,
    const unsigned int s,
    const unsigned int width, //Of upsampled
    const unsigned int height)//Of upsampled
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int factor = (int) pow(2.f, (float)s);
    int offset = x+y*width;
    int fwidth = width/factor;
    int fheight = height/factor;

    out_tex[offset] = in_tex[(x/factor)+(y/factor)*(width/factor)]; //truncated
}

//NOTE: not used
__kernel void mip_reduce( //not the best
    image2d_t in_tex,
    __write_only image2d_t out_tex,
    const unsigned int width, //Of reduced
    const unsigned int height)//Of reduced
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;



    vec4 p00 = read_imagef(in_tex, sampler, (int2)(x*2,   y*2  ));

    vec4 p01 = read_imagef(in_tex, sampler, (int2)(x*2+1, y*2  ));

    vec4 p10 = read_imagef(in_tex, sampler, (int2)(x*2,   y*2+1));

    vec4 p11 = read_imagef(in_tex, sampler, (int2)(x*2+1, y*2+1));

    write_imagef(out_tex, (int2)(x,y), p00+p01+p10+p11/4.f);
}
