#pragma once
#include <alignment_util.h>
#include <stdbool.h>
#include <stdint.h>

typedef int   ivec3[4]; //1 int padding
typedef float vec2[2];
typedef float vec3[4];  //1 float padding
typedef float vec4[4];
typedef float mat4[16];

/*******/
/* Ray */
/*******/
typedef struct ray
{
    vec3 orig;
    vec3 dir;

} ray; //already aligned


/**************/
/* Voxel/AABB */
/**************/

typedef struct AABB
{
    vec3 max;
    vec3 min;
} AABB;

void  AABB_divide(AABB, uint8_t, float, AABB*, AABB*);
void  AABB_divide_world(AABB, uint8_t,  float, AABB*, AABB*);
float AABB_surface_area(AABB);
void  AABB_clip(AABB*, AABB*, AABB*);
float AABB_ilerp(AABB, uint8_t, float);
bool  AABB_is_planar(AABB*, uint8_t);

void  AABB_construct_from_vertices(AABB*, vec3*,  unsigned int);
void  AABB_construct_from_triangle(AABB*, ivec3*, vec3*);


/**********/
/* Sphere */
/**********/

//NOTE:  less memory efficient but aligns with OpenCL
typedef W_ALIGN(16) struct //sphere
{
    //GPU stores all vec3s as vec4s in memory so we need the padding.
    vec4 pos;

    float radius;
    int material_index;

}  U_ALIGN(16) sphere;


float does_collide_sphere(sphere, ray);


/*********/
/* Plane */
/*********/

typedef W_ALIGN(16) struct plane
{
    vec4 pos;
    vec4 norm;

    //32 bytes by here

    int material_index;
} U_ALIGN(16) plane;
float does_collide_plane(plane, ray);

ray generate_ray(int x, int y, int width, int height, float fov);
float* matvec_mul(mat4 m, vec4 v);
