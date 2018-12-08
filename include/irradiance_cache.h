#pragma once
#include <stdint.h>

#define NUM_MIPMAPS 4 //NOTE: 1080/(2^4) != integer

typedef struct _rt_ctx raytracer_context;

//    0 = 000: x-, y-, z-
//    1 = 001: x-, y-, z+
//    2 = 010: x-, y+, z-
//    3 = 011: x-, y+, z+
//    4 = 100: x+, y-, z-
//    5 = 101: x+, y-, z+
//    6 = 110: x+, y+, z-
//    7 = 111: x+, y+, z+

typedef struct
{
    vec3 point;
    vec3 normal;

    float rad;

    vec3 col;

    vec3 gpos;
    vec3 gdir;
} ic_ir_value;

typedef struct _ic_octree_node ic_octree_node;

struct _ic_octree_node
{
    bool leaf;
    bool active;

    union
    {
        struct
        {
            unsigned int buffer_offset;
            unsigned int num_elems;
        } leaf;
        struct
        {

            ic_octree_node* children[8];
        } branch;
    } data;
    vec3 min;
    vec3 max;
};

typedef struct
{
    ic_octree_node* root;
    int node_count;
    unsigned int width;
    unsigned int max_depth;
} ic_octree;

typedef struct
{
    //vec4* texture;
    cl_mem cl_image_ref;
    unsigned int width, height;
}  ic_mipmap_gb;

typedef struct
{
    //float* texture;
    cl_mem cl_image_ref;
    unsigned int width, height;
}  ic_mipmap_f;

typedef struct
{

    cl_image_format cl_standard_format;
    cl_image_desc   cl_standard_descriptor;
    ic_octree       octree;
    ic_ir_value*    ir_buf;
    unsigned int ir_buf_size;
    unsigned int ir_buf_current_offset;
}  ic_context;

void ic_init(raytracer_context*);
void ic_screenspace(raytracer_context*);
void ic_octree_init_branch(ic_octree_node*);
void ic_octree_insert(ic_context*, vec3 point, vec3 normal);
