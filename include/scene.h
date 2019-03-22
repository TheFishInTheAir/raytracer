#pragma once
#include <alignment_util.h>
#include <vec.h>
//typedef struct{} sphere;
//struct sphere;
//struct plane;
//struct kd_tree;

typedef struct _rt_ctx raytracer_context;

typedef W_ALIGN(16) struct
{
    vec4  colour;

    float reflectivity;

    //TODO: add more.
} U_ALIGN(16) material;



typedef W_ALIGN(32) struct
{
    mat4 model;

    vec4 max;
    vec4 min;

    int index_offset;
    int num_indices;

    int material_index;
} U_ALIGN(32) mesh;

typedef struct
{

    mat4 camera_world_matrix;

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
    bool spheres_changed;
    //Planes
    plane* planes;
    cl_mem cl_plane_buffer;
    unsigned int num_planes; //NOTE: must be constant.
    bool planes_changed;

    //Meshes
    mesh* meshes; //All vertex data is stored contiguously
    cl_mem cl_mesh_buffer;
    unsigned int num_meshes;
    bool meshes_changed;

    //Trying to remember how I got all of the other structs to use typedefs...
    //kd_tree
    struct kd_tree* kdt;


    //NOTE: we could store vertices, normals, and texcoords contiguously as 1 buffer.
    vec3* mesh_verts;
    cl_mem cl_mesh_vert_buffer;
    unsigned int num_mesh_verts; //NOTE: must be constant.

    vec3* mesh_nrmls;
    cl_mem cl_mesh_nrml_buffer;
    unsigned int num_mesh_nrmls; //NOTE: must be constant.

    vec2* mesh_texcoords;
    cl_mem cl_mesh_texcoord_buffer;
    unsigned int num_mesh_texcoords; //NOTE: must be constant.

    ivec3* mesh_indices;
    cl_mem cl_mesh_index_buffer;
    unsigned int num_mesh_indices; //NOTE: must be constant.

} scene;


void scene_resource_push(raytracer_context*);
void scene_init_resources(raytracer_context*);
void scene_generate_resources(raytracer_context*); //k-d tree generation
