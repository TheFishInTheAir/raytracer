//#define SET_PIXEL(x,y, tex, result)
#define FOV 80.0f

#define vec3 float3
#define vec4 float4
#define TYPE_SPHERE 0
#define TYPE_PLANE  1




/************/
/* Material */
/************/
typedef struct
{
    float reflectivity;
    vec3 colour;
} material;

//TODO: refactor var names
material get_material(__global float* buf, int offset) //NOTE: offset is index (woule be a better name)
{
    int real_offset = offset*(4);

    material m;

    m.reflectivity = buf[0 + real_offset];
    m.colour.x     = buf[1 + real_offset];
    m.colour.y     = buf[2 + real_offset];
    m.colour.z     = buf[3 + real_offset];

    return m;
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

    int material_index;
} sphere;
sphere get_sphere(__global float* sphere_data, int offset)
{
    int real_offset = offset*(5);


    sphere s;
    s.pos.x  = sphere_data[0 + real_offset];
    s.pos.y  = sphere_data[1 + real_offset];
    s.pos.z  = sphere_data[2 + real_offset];

    s.radius = sphere_data[3 + real_offset];

    s.material_index = ((__global int*)sphere_data)[4 + real_offset];

    //s.pos.x = 0.0f;
    //s.pos.y = 0.0f;
    //s.pos.z = -5.0f;
    //s.radius = 0.5f;
    return s;
}

/*********/
/* Plane */
/*********/

typedef struct plane
{
    vec3 pos;
    vec3 norm;

    int material_index;
} plane;
plane get_plane(__global float* plane_data, int offset)
{
    int real_offset = offset*(7);

    plane p;
    p.pos.x  = plane_data[0 + real_offset];
    p.pos.y  = plane_data[1 + real_offset];
    p.pos.z  = plane_data[2 + real_offset];

    p.norm.x = plane_data[3 + real_offset];
    p.norm.y = plane_data[4 + real_offset];
    p.norm.z = plane_data[5 + real_offset];

    p.material_index = ((__global int*)plane_data)[6 + real_offset];

    return p;
}


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


//OTHER THING
typedef struct
{
    bool did_hit;
    vec3 norm;
    vec3 point;
    material mat;
    //TODO: Add material
} collision_result;


/*************/
/* Collision */
/*************/
void swap_float(float *f1, float *f2)
{
    float temp = *f2;
    *f2 = *f1;
    *f1 = temp;
}

float does_collide_sphere(sphere s, ray r)
{
    float t0, t1; // solutions for t if the ray intersects

    // analytic solution
    vec3 L = s.pos- r.orig;
    float b = dot(r.dir, L) ;//* 2.0f;
    float c = dot(L, L) - (s.radius*s.radius); //NOTE: you can optimize out the square.

    float disc = b * b - c/**a*/; /* discriminant of quadratic formula */

    /* solve for t (distance to hitpoint along ray) */
    float t = -1.0f;

    if (disc < 0.0f) return -1.0f;
    else t = b - sqrt(disc);

    if (t < 0.0f)
    {
        t = b + sqrt(disc);
        if (t < 0.0f) return -1.0f;
    }
    return t;
}

float does_collide_plane(plane p, ray r)
{
    float denom = dot(r.dir, p.norm);
    if (fabs(denom) > 1e-6)
    {
        vec3 l = p.pos - r.orig;
        float t = dot(l, p.norm) / denom;
        if (t >= 0)
            return t;

    }
    return -1.0;
}

vec3 reflect(vec3 incidentVec, vec3 normal)
{
    return incidentVec - 2.f * dot(incidentVec, normal) * normal;
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


#define TEMP_FAR_PLANE 100000000

collision_result collide(ray r,
                         __global float* materials,
                         __global float* spheres,
                         __global float* planes)
{
    collision_result result;
    result.did_hit = false;
    result.point = (vec3)(0);
    result.norm = (vec3)(0);

    float dist = TEMP_FAR_PLANE; //far plane
    unsigned char collision_type = 0;
    vec3 colour;
    sphere s;
    plane p;

    for(int i = 0; i < SCENE_NUM_SPHERES; i++)
    {
        sphere current_sphere = get_sphere(spheres, i);
        float local_dist = does_collide_sphere(current_sphere, r);
        if(local_dist<dist && local_dist!=-1)
        {
            dist = local_dist;
            s = current_sphere;
            collision_type = TYPE_SPHERE;
            colour = (vec3)((float)((i+1)*17%3)/3.0f, (float)((i+1)*16%5)/5.0f, (float)((i+1)*13%7)/7.0f);
        }
        //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.
    }

    for(int i = 0; i < SCENE_NUM_PLANES; i++)
    {
        plane current_plane = get_plane(planes, i);
        float local_dist = does_collide_plane(current_plane, r);
        if(local_dist<dist && local_dist!=-1)
        {
            dist = local_dist;
            p = current_plane;
            collision_type = TYPE_PLANE;
            colour = (vec3)(1.0f);

        }

        //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.
    }

    if(dist!=TEMP_FAR_PLANE)
    {
        int mat_index;
        vec3 normal;
        vec3 point = r.orig + (r.dir * dist);
        switch(collision_type)
        {
        case TYPE_SPHERE:
        {
            normal = normalize(point - s.pos);
            mat_index = s.material_index;
        }break;
        case TYPE_PLANE:
        {
            normal = p.norm;
            mat_index = p.material_index;
        }break;
        }

        result.did_hit = true;
        result.point = point;
        result.norm = normal;
        result.mat = get_material(materials, mat_index);

    }

    return result;

}

vec4 shade(collision_result result) //NOTE: Temp shitty phong
{

    const vec3 light_pos = (vec3)(2,5,-1);
    vec3 nspace_light_dir = normalize(light_pos-result.point);
    vec4 test_lighting = (vec4) (clamp((float)dot(result.norm, nspace_light_dir), 0.0f, 1.0f));
    test_lighting *= (vec4)(result.mat.colour, 1.0f);
    return test_lighting;
}


#define REFLECTIVITY 0.4f;
__kernel void cast_ray_test( //TODO: optimize global memory access.
    __global unsigned int* out_tex,
    const __global float* ray_buffer,
    const __global float* material_buffer,
    const __global float* spheres,
    const __global float* planes,
    const unsigned int width,
    const unsigned int height)
{
    const vec4 sky = (vec4) (0.2, 0.8, 0.5, 0);
    //return;
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
    collision_result result = collide(r, material_buffer, spheres, planes);
    vec4 colour = shade(result);

    ray secondary_ray;
    secondary_ray.orig = result.point + result.norm * 0.0001f; //NOTE: BIAS
    secondary_ray.dir  = reflect(r.dir, result.norm);//reflect(r.dir, result.norm);
    collision_result result2 = collide(secondary_ray, material_buffer, spheres, planes);
    vec4 reflected_colour = shade(result2);

    ray secondary_ray2;
    secondary_ray2.orig = result2.point + result2.norm * 0.0001f; //NOTE: BIAS
    secondary_ray2.dir  = reflect(secondary_ray.dir, result2.norm);//reflect(r.dir, result.norm);
    collision_result result22 = collide(secondary_ray2, material_buffer, spheres, planes);
    vec4 reflected_colour2 = shade(result22);
/*
    ray secondary_ray22;
    secondary_ray22.orig = result22.point + result22.norm * 0.0001f; //NOTE: BIAS
    secondary_ray22.dir  = reflect(secondary_ray2.dir, result22.norm);//reflect(r.dir, result.norm);
    collision_result result222 = collide(secondary_ray22, spheres, planes);
    vec4 reflected_colour22 = shade(result222);
    colour *= 1-REFLECTIVITY;
    reflected_colour *= 1-REFLECTIVITY;
    reflected_colour2 *= 1-REFLECTIVITY;
    reflected_colour22 *= 1-REFLECTIVITY;

    if(result222.did_hit)
    {
        reflected_colour2 += reflected_colour22*REFLECTIVITY;
        }*/
    //if(!result22.did_hit)
    //    reflected_colour2 = sky;

    if(result22.did_hit)
    {
        reflected_colour2 = mix(reflected_colour2, (vec4)(0,0,0,0), result22.mat.reflectivity);
    }
    else
    {
        reflected_colour2 = sky;
    }


    if(result2.did_hit)
    {
        reflected_colour = mix(reflected_colour, reflected_colour2, result2.mat.reflectivity);
    }
    else
    {
        reflected_colour = sky;
    }

    //reflected_colour2 = mix(reflected_colour2, (vec4)(0,0,0,0), result22.mat.reflectivity);

    //reflected_colour = mix(reflected_colour, reflected_colour2, result2.mat.reflectivity);

    colour = mix(colour, reflected_colour,  result.mat.reflectivity);

    //out_tex[x+y*width] = getColour( (vec4)(0.8,0.3,dist/10,0));
    if(result.did_hit) //actually doing this is faster then an early exit....
    {
        out_tex[x+y*width] = get_colour( colour );
    }
    else
    {
        out_tex[x+y*width] = get_colour( sky );
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
