
vec4 shade(collision_result result, scene s, MESH_SCENE_DATA_PARAM)
{
    const vec3 light_pos = (vec3)(1,2, 0);
    vec3 nspace_light_dir = normalize(light_pos-result.point);
    vec4 test_lighting ;//= (vec4) (clamp((float)dot(result.normal, nspace_light_dir), 0.0f, 1.0f));
    ray r;
    r.dir  = nspace_light_dir;
    r.orig = result.point + nspace_light_dir*0.00001f;
    collision_result _cr;
    bool visible = !collide_all(r, &_cr, s, MESH_SCENE_DATA);
    test_lighting = (vec4)(result.mat.colour, 1.0f);
    return visible*test_lighting;
}


__kernel void cast_ray_test(
    __global unsigned int* out_tex,
    const __global ray* ray_buffer,
    const __global material* material_buffer,
    const __global sphere* spheres,
    const __global plane* planes,
//Mesh
    const __global mesh* meshes,
    image1d_buffer_t indices,
    image1d_buffer_t vertices,
    image1d_buffer_t normals,
    /* const __global vec2* texcoords, */
    /* , */


    const unsigned int width,
    const unsigned int height,
    const vec4 pos)
{
    scene s;
    s.material_buffer = material_buffer;
    s.spheres         = spheres;
    s.planes          = planes;
    s.meshes          = meshes;

    const vec4 sky = (vec4) (0.84, 0.87, 0.93, 0);
    //return;
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;
    int ray_offset = offset;


    ray r;
    r = ray_buffer[ray_offset];
    //r.orig = pos.xyz; //NOTE: unnecesesary rn, in progress of updating kernels w/ the new ray buffers.

    //r.dir  = (vec3)(0,0,-1);

    //out_tex[x+y*width] = get_colour_signed((vec4)(r.dir,0));
    //out_tex[x+y*width] = get_colour_signed((vec4)(1,1,0,0));
    collision_result result;
    if(!collide_all(r, &result, s, MESH_SCENE_DATA))
    {
        out_tex[x+y*width] = get_colour( sky );
        return;
    }
    vec4 colour = shade(result, s, MESH_SCENE_DATA);


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
            rays[i].dir  = reflect(r.dir, result.normal);
        }
        else
        {
            rays[i].orig = results[i-1].point + results[i-1].normal * 0.0001f; //NOTE: BIAS
            rays[i].dir  = reflect(rays[i-1].dir, results[i-1].normal);
        }
        if(collide_all(rays[i], results+i, s, MESH_SCENE_DATA))
        {
            colours[i] = shade(results[i], s, MESH_SCENE_DATA);
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
        if(i==NUM_REFLECTIONS-1)
            colours[i] = mix(colours[i], sky, results[i].mat.reflectivity);

        else
            colours[i] = mix(colours[i], colours[i+1], results[i].mat.reflectivity);

    }

    colour = mix(colour, colours[0],  result.mat.reflectivity);
#ifdef WIN32
    out_tex[offset] = get_colour( colour );
#else
    out_tex[offset] = get_colour( colour.zyxw );
#endif
}


//NOTE: it might be faster to make the ray buffer a multiple of 4 just to align with words...
__kernel void generate_rays(
    __global ray* out_tex,
    const unsigned int width,
    const unsigned int height,
    const t_mat4 wcm)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width);

    ray r;

    float aspect_ratio = width / (float)height; // assuming width > height
    float cam_x = (2 * (((float)x + 0.5) / width) - 1) * tan(FOV / 2 * M_PI_F / 180) * aspect_ratio;
    float cam_y = (1 - 2 * (((float)y + 0.5) / height)) * tan(FOV / 2 * M_PI_F / 180);

    //r.orig = matvec((float*)&wcm, (vec4)(0.0, 0.0, 0.0, 1.0)).xyz;
    //r.dir  = matvec((float*)&wcm, (vec4)(cam_x, cam_y, -1.0f, 1)).xyz - r.orig;

    r.orig = (vec3)(0, 0, 0);
    r.dir  = (vec3)(cam_x, cam_y, -1.0f) - r.orig;

    r.dir = normalize(r.dir);

    out_tex[offset]   = r;
}
