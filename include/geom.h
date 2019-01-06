#pragma once
#include <alignment_util.h>
#include <stdbool.h>

typedef int   ivec3[4]; //1 int padding
typedef float vec2[2];
typedef float vec3[4]; //1 float padding
typedef float vec4[4];
typedef float mat4[16];

/*******/
/* Ray */
/*******/
typedef struct ray
{
    vec3 orig;
    vec3 dir;
    //float t_min, t_max;
} ray;


/**********/
/* Sphere */
/**********/

//NOTE:  less memory efficient but aligns with opencl
typedef W_ALIGN(16) struct //sphere
{
    vec4 pos; //GPU stores all vec3s as vec4s in memory so we need the padding.

    float radius;
    int material_index;

}  U_ALIGN(16) sphere;


float does_collide_sphere(sphere, ray);


/*********/
/* Plane */
/*********/

typedef W_ALIGN(16) struct plane // bytes
{
    vec4 pos; //12
    //float test;

    vec4 norm;
    //float test2;

//32
    int material_index;
} U_ALIGN(16) plane;
float does_collide_plane(plane, ray);

ray generate_ray(int x, int y, int width, int height, float fov);
float* matvec_mul(mat4 m, vec4 v);
