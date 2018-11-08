#pragma once
#include <stdbool.h>


typedef float vec3[3];
typedef float vec4[4];

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

typedef struct sphere
{
    vec3 pos;
    float radius;
} sphere;
float does_collide_sphere(sphere, ray);


/*********/
/* Plane */
/*********/

typedef struct plane
{
    vec3 pos;
    vec3 norm;
} plane;
float does_collide_plane(plane, ray);

ray generate_ray(int x, int y, int width, int height, float fov);
