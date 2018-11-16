//#define SET_PIXEL(x,y, tex, result)
#define FOV 80.0f

#define vec3 float3
#define vec4 float4


/*******/
/* Ray */
/*******/
typedef struct
{
    vec3 orig;
    vec3 dir;
} ray;

ray generate_ray(int x, int y, int width, int height, float fov)
{
    ray r;

    float aspect_ratio = width / (float)height; // assuming width > height
    float cam_x = (2 * (((float)x + 0.5) / width) - 1) * tan(fov / 2 * M_PI / 180) * aspect_ratio;
    float cam_y = (1 - 2 * (((float)y + 0.5) / height)) * tan(fov / 2 * M_PI / 180);

    r.orig = (0);
    r.dir  = (vec3)(cam_x, cam_y, -1.0f) - r.orig;
    r.dir = normalize(r.dir);

    return r;
}


/**********************/
/*                    */
/*     Primitives     */
/*                    */
/**********************/

/**********/
/* Sphere */
/**********/
typedef struct
{
    vec3 pos;
    float radius;
} sphere;
sphere get_sphere(__global float* sphere_data, int offset)
{
    sphere s;
    s.pos.x  = sphere_data[0 + (offset*4)];
    s.pos.y  = sphere_data[1 + (offset*4)];
    s.pos.z  = sphere_data[2 + (offset*4)];
    s.radius = sphere_data[3 + (offset*4)];
    //s.pos.x = 0.0f;
    //s.pos.y = 0.0f;
    //s.pos.z = -5.0f;
    //s.radius = 0.5f;
    return s;
}


/*************/
/* Collision */
/*************/
void swap_float(float *f1, float *f2)
{
    float temp = *f2;
    *f2 = *f1;
    *f1 = temp;
}

bool solve_quadratic(float *a, float *b, float *c, float *x0, float *x1)
{
    float discr = (*b) * (*b) - 4 * (*a) * (*c);
    // printf("test: %f    a:%f, b:%f, c:%f\n", discr, *a, *b, *c);
    if (discr < 0.0f) return false;
    else if (discr == 0.0f) {
        (*x0) = (*x1) = - 0.5f * (*b) / (*a);
    }
    else {
        float q = (*b > 0) ?
            -0.5 * (*b + sqrt(discr)) :
            -0.5 * (*b - sqrt(discr));
        *x0 = q / *a;
        *x1 = *c / q;
    }

    return true;
}

float does_collide_sphere(sphere s, ray r)
{
    float t0, t1; // solutions for t if the ray intersects

    // analytic solution
    vec3 L = r.orig-s.pos;
    float a = 1.0f; //NOTE: we always normalize the direction vector.
    float b = dot(r.dir, L) * 2.0f;
    float c = dot(L, L) - (s.radius*s.radius); //NOTE: you can optimize out the square.
    if (!solve_quadratic(&a, &b, &c, &t0, &t1)) return -1.0f;

    if (t0 > t1) swap_float(&t0, &t1);

    if (t0 < 0) {
        t0 = t1; // if t0 is negative, let's use t1 instead
        if (t0 < 0) return -1.0f; // both t0 and t1 are negative
    }


    return t0;
}

unsigned int get_colour(vec4 col)
{
    unsigned int outCol = 0;

    col = clamp(col, 0.0f, 1.0f);

    outCol |= 0xff000000 & (unsigned int)(col.w*255)<<24;
    outCol |= 0x00ff0000 & (unsigned int)(col.x*255)<<16;
    outCol |= 0x0000ff00 & (unsigned int)(col.y*255)<<8;
    outCol |= 0x000000ff & (unsigned int)(col.z*255);


    /* outCol |= 0xff000000 & min((unsigned int)(col.w*255), (unsigned int)255)<<24; */
    /* outCol |= 0x00ff0000 & min((unsigned int)(col.x*255), (unsigned int)255)<<16; */
    /* outCol |= 0x0000ff00 & min((unsigned int)(col.y*255), (unsigned int)255)<<8; */
    /* outCol |= 0x000000ff & min((unsigned int)(col.z*255), (unsigned int)255); */
    return outCol;
    //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.

}


vec4 get_from_colour(unsigned int raw)
{
    vec4 col;

    col.w = (float)((raw & 0xff000000) >> 24) / 255.0f;
    col.x = (float)((raw & 0x00ff0000) >> 16) / 255.0f;
    col.y = (float)((raw & 0x0000ff00) >> 8)  / 255.0f;
    col.z = (float)(raw & 0x000000ff)  / 255.0f;

    return col;
}

unsigned int get_colour_signed(vec4 col) //NOTE: THIS IS TRUNCATING FLOATS
{
    unsigned int out_col = 0;
    char* out_col_ref = (char*) &out_col;
    //col /= 2;
    //col += 0.5f;

    //col = clamp(col, -1.0f, 1.0f);

    *(out_col_ref)   = (char) (col.z*126);
    *(out_col_ref+1) = (char) (col.y*126);
    *(out_col_ref+2) = (char) (col.x*126);
    *(out_col_ref+3) = (char) (col.w*126);
/*
    outCol |= 0xff000000 & ((char)(col.w*255)|0x00000000)<<24;
    outCol |= 0x00ff0000 & ((char)(col.x*255)|0x00000000)<<16;
    outCol |= 0x0000ff00 & ((char)(col.y*255)|0x00000000)<<8;
    outCol |= 0x000000ff & (char)(col.z*255);
*/

    /* outCol |= 0xff000000 & min((unsigned int)(col.w*255), (unsigned int)255)<<24; */
    /* outCol |= 0x00ff0000 & min((unsigned int)(col.x*255), (unsigned int)255)<<16; */
    /* outCol |= 0x0000ff00 & min((unsigned int)(col.y*255), (unsigned int)255)<<8; */
    /* outCol |= 0x000000ff & min((unsigned int)(col.z*255), (unsigned int)255); */
    return out_col;
}

vec4 get_from_colour_signed(unsigned int raw)
{
    vec4 col;

    char* raw_ref = (char*) &raw;

    col.z = ((float)*(raw_ref+0))/126.0f;
    col.y = ((float)*(raw_ref+1))/126.0f;
    col.x = ((float)*(raw_ref+2))/126.0f;
    col.w = ((float)*(raw_ref+3))/126.0f;

    /* col.w = (float)((raw & 0xff000000) >> 24) / 255.0f; */
    /* col.x = (float)((raw & 0x00ff0000) >> 16) / 255.0f; */
    /* col.y = (float)((raw & 0x0000ff00) >> 8)  / 255.0f; */
    /* col.z = (float)(raw & 0x000000ff)  / 255.0f; */

    //col -= 0.5f;
    //col *= 2;
    return col;
    //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.

}

/*
__kernel void magenta_test(
    __global unsigned int* out_tex,
    __global float* spheres,
    const unsigned int width,
    const unsigned int height)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;

    //sphere s = get_sphere(spheres); //Get forst sphere

    ray r = generate_ray(x,y, width, height, FOV);


    float dist = -1;
    sphere s;

    for(int i = 0; i < SCENE_NUM_SPHERES; i++)
    {
        //if(i >= count)
        //    continue;
        sphere current_sphere = get_sphere(spheres, i);

        float local_dist = does_collide_sphere(current_sphere, r);
        if(local_dist==-1.0f)
            continue;

        if(local_dist<dist || dist == -1.0f)
        {
            dist = local_dist;
            s = current_sphere;
        }

        //out_tex[x+y*width] = sphere_light_calc(get_sphere(current_sphere), r);
        //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.
    }

    if(dist!=-1)
    {
        vec3 p_hit = r.orig + (r.dir * dist); // O+tD
        vec3 normal = normalize(p_hit - s.pos);

        vec3 light_pos = (vec3)(2,5,-1);
        vec3 nspace_light_dir = normalize(light_pos-s.pos);

        float test_lighting = clamp((float)dot(normal, nspace_light_dir), 0.0f, 1.0f);
        //out_tex[x+y*width] = getColour( (vec4)(0.8,0.3,dist/10,0));
        out_tex[x+y*width] = get_colour( (vec4)(test_lighting));
    }
    else
    {
        out_tex[x+y*width] = get_colour( (vec4)(0.2,0.8,0.5,0));
        //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.
    }
}*/

__kernel void cast_ray_test(
    __global unsigned int* out_tex,
    __global float* ray_buffer,
    __global float* spheres,
    const unsigned int width,
    const unsigned int height)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;
    int ray_offset = offset*3;

    ray r;
    r.orig = (vec3)(0,0,0);
    //r.dir  = (vec3)(get_from_colour_signed(ray_buffer[offset]).zyz);
    r.dir.x = ray_buffer[ray_offset];
    r.dir.y = ray_buffer[ray_offset+1];
    r.dir.z = ray_buffer[ray_offset+2];

    //r.dir  = (vec3)(0,0,-1);

    //out_tex[x+y*width] = get_colour_signed((vec4)(r.dir,0));
    //out_tex[x+y*width] = get_colour_signed((vec4)(1,1,0,0));

    //return;
    float dist = -1;
    sphere s;

    for(int i = 0; i < SCENE_NUM_SPHERES; i++)
    {
        //if(i >= count)
        //    continue;
        sphere current_sphere = get_sphere(spheres, i);

        float local_dist = does_collide_sphere(current_sphere, r);
        if(local_dist==-1.0f)
            continue;

        if(local_dist<dist || dist == -1.0f)
        {
            dist = local_dist;
            s = current_sphere;
        }

        //out_tex[x+y*width] = sphere_light_calc(get_sphere(current_sphere), r);
        //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.
    }

    if(dist!=-1)
    {
        vec3 p_hit = r.orig + (r.dir * dist); // O+tD
        vec3 normal = normalize(p_hit - s.pos);

        vec3 light_pos = (vec3)(2,5,-1);
        vec3 nspace_light_dir = normalize(light_pos-s.pos);

        float test_lighting = clamp((float)dot(normal, nspace_light_dir), 0.0f, 1.0f);
        //out_tex[x+y*width] = getColour( (vec4)(0.8,0.3,dist/10,0));
        out_tex[x+y*width] = get_colour( (vec4)(test_lighting));
    }
    else
    {
        out_tex[x+y*width] = get_colour( (vec4)(0.2,0.8,0.5,0));
        //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.
    }
    //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.

}

//NOTE: it might be faster to make the ray buffer a multiple of 4 just to fit word size...
__kernel void generate_rays(
    __global float* out_tex,
    const unsigned int width,
    const unsigned int height)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width) * 3;



    ray r = generate_ray(x,y, width, height, FOV);
    out_tex[offset] = r.dir.x;
    out_tex[offset+1] = r.dir.y;
    out_tex[offset+2] = r.dir.z;



    //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.

}
