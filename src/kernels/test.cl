
vec4 shade(collision_result result) //NOTE: Temp shitty phong
{
    const vec3 light_pos = (vec3)(2,5,-1);
    vec3 nspace_light_dir = normalize(light_pos-result.point);
    vec4 test_lighting = (vec4) (clamp((float)dot(result.normal, nspace_light_dir), 0.0f, 1.0f));
    test_lighting *= (vec4)(result.mat.colour, 1.0f);
    return test_lighting;
}


__kernel void cast_ray_test(
    __global unsigned int* out_tex,
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
    const vec4 pos)
{
    const vec4 sky = (vec4) (0.2, 0.8, 0.5, 0);
    //return;
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;
    int ray_offset = offset*3;


    ray r;
    r.orig = pos.xyz; //NOTE: unoptimized unaligned memory access.
    r.dir.x = ray_buffer[ray_offset];
    r.dir.y = ray_buffer[ray_offset+1];
    r.dir.z = ray_buffer[ray_offset+2];

    //r.dir  = (vec3)(0,0,-1);

    //out_tex[x+y*width] = get_colour_signed((vec4)(r.dir,0));
    //out_tex[x+y*width] = get_colour_signed((vec4)(1,1,0,0));
    collision_result result;
    if(!collide_all(r, &result, material_buffer, spheres, planes, meshes, indices,
                   vertices, normals))
    {
        out_tex[x+y*width] = get_colour( sky );
        return;
    }
    vec4 colour = shade(result);


    #define NUM_REFLECTIONS 2
    ray rays[NUM_REFLECTIONS];
    collision_result results[NUM_REFLECTIONS];
    vec4 colours[NUM_REFLECTIONS];
    int early_exit_num = NUM_REFLECTIONS;
    for(int i = 0; i < NUM_REFLECTIONS; i++)
    {
        if(i==0)
        {
            rays[i].orig = result.point + result.normal * 0.0001f; //NOTE: BIAS
            rays[i].dir  = reflect(r.dir, result.normal);//reflect(r.dir, result.norm);
        }
        else
        {
            rays[i].orig = results[i-1].point + results[i-1].normal * 0.0001f; //NOTE: BIAS
            rays[i].dir  = reflect(rays[i-1].dir, results[i-1].normal);//reflect(r.dir, result.norm);
        }
        if(collide_all(rays[i], results+i, material_buffer, spheres,
                       planes, meshes, indices, vertices, normals))
        {
            colours[i] = shade(results[i]);
        }
        else
        {
            colours[i] = sky;
            early_exit_num = i;
            break;
        }
    }
    for(int i = early_exit_num-1; i > -1; i--)
    {
        //if(i==0)
            //{
        if(i==NUM_REFLECTIONS-1)
            colours[i] = mix(colours[i], sky, results[i].mat.reflectivity);

        else
            colours[i] = mix(colours[i], colours[i+1], results[i].mat.reflectivity);


            //}
            //else
            //{

            //}
    }

    colour = mix(colour, colours[0],  result.mat.reflectivity);

    out_tex[x+y*width] = get_colour( colour );

    //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.

}


//NOTE: it might be faster to make the ray buffer a multiple of 4 just to fit word size...
__kernel void generate_rays(
    __global float* out_tex,
    const unsigned int width,
    const unsigned int height,
    const t_mat4 wcm)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width) * 3;

    ray r;

    float aspect_ratio = width / (float)height; // assuming width > height
    float cam_x = (2 * (((float)x + 0.5) / width) - 1) * tan(FOV / 2 * M_PI / 180) * aspect_ratio;
    float cam_y = (1 - 2 * (((float)y + 0.5) / height)) * tan(FOV / 2 * M_PI / 180);

    //r.orig = matvec((float*)&wcm, (vec4)(0.0, 0.0, 0.0, 1.0)).xyz;
    //r.dir  = matvec((float*)&wcm, (vec4)(cam_x, cam_y, -1.0f, 1)).xyz - r.orig;

    r.orig = (vec3)(0, 0, 0);
    r.dir  = (vec3)(cam_x, cam_y, -1.0f) - r.orig;

    r.dir = normalize(r.dir);

    out_tex[offset]   = r.dir.x;
    out_tex[offset+1] = r.dir.y;
    out_tex[offset+2] = r.dir.z;
}
