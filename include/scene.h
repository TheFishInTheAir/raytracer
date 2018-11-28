#pragma once
#include <vec.h>
//typedef struct{} sphere;
struct sphere;
struct plane;
typedef struct _rt_ctx raytracer_context; //hallelujah?...

typedef struct
{
    float reflectivity;
    vec3  colour;
    //TODO: add more.
} material;

typedef struct
{
    vec3 max;
    vec3 min;
} AABB;

typedef struct
{
    int vertex_offset;
    int num_verticies;

    int normal_offset;
    int num_normals;

    int index_offset;
    int num_indices;

    mat4 model;

    AABB aabb;

    int material_index;
} mesh;

typedef struct
{
    //Materials
    material* materials;
    cl_mem cl_material_buffer;
    unsigned int num_materials;
    bool materials_changed;
    //Primatives

    //Spheres
    sphere* spheres;
    cl_mem cl_sphere_buffer;
    unsigned int num_spheres; //NOTE: must be constant.
    bool spheres_changed; //should re-push values.
    //Planes
    plane* planes;
    cl_mem cl_plane_buffer;
    unsigned int num_planes; //NOTE: must be constant.
    bool planes_changed; //should re-push values.

    //Meshes
    mesh* meshes; //All vertex data is stored contiguously
    cl_mem cl_mesh_buffer;
    unsigned int num_meshes; //NOTE: must be constant.
    bool meshes_changed;

    vec3* mesh_verts;
    cl_mem cl_mesh_vert_buffer;
    unsigned int num_mesh_verts; //NOTE: must be constant.

    vec3* mesh_nrmls;
    cl_mem cl_mesh_nrml_buffer;
    unsigned int num_mesh_nrmls; //NOTE: must be constant.

    vec3* mesh_indices;
    cl_mem cl_mesh_index_buffer;
    unsigned int num_mesh_indices; //NOTE: must be constant.

} scene;


void scene_resource_push(raytracer_context*);
void scene_init_resources(raytracer_context*);
