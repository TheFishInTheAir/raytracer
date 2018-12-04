#include <scene.h>
#include <raytracer.h>

#include <geom.h>
#include <CL/cl.h>

void scene_init_resources(raytracer_context* rctx)
{
    int err;

    //Scene Buffers
    rctx->stat_scene->cl_sphere_buffer = clCreateBuffer(rctx->rcl->context,
                                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                        sizeof(plane)*rctx->stat_scene->num_spheres,
                                                        rctx->stat_scene->spheres, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Scene Sphere Buffer. %i\n", err);
        exit(1);
    }

    rctx->stat_scene->cl_plane_buffer = clCreateBuffer(rctx->rcl->context,
                                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                       sizeof(plane)*rctx->stat_scene->num_planes,
                                                       rctx->stat_scene->planes, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Scene Plane Buffer. %i\n", err);
        exit(1);
    }

    rctx->stat_scene->cl_material_buffer = clCreateBuffer(rctx->rcl->context,
                                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                          sizeof(material)*rctx->stat_scene->num_materials,
                                                          rctx->stat_scene->materials, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Scene Material Buffer. %i\n", err);
        exit(1);
    }

    //Mesh
    rctx->stat_scene->cl_mesh_buffer = clCreateBuffer(rctx->rcl->context,
                                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                      sizeof(mesh)*rctx->stat_scene->num_meshes,
                                                      rctx->stat_scene->meshes, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Scene Mesh Buffer. %i\n", err);
        exit(1);
    }
    rctx->stat_scene->cl_mesh_vert_buffer = clCreateBuffer(rctx->rcl->context,
                                                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                          sizeof(vec3)*rctx->stat_scene->num_mesh_verts,
                                                          rctx->stat_scene->mesh_verts, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Scene Mesh Vertex Buffer. %i\n", err);
        exit(1);
    }
    rctx->stat_scene->cl_mesh_nrml_buffer = clCreateBuffer(rctx->rcl->context,
                                                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                           sizeof(vec3)*rctx->stat_scene->num_mesh_nrmls,
                                                           rctx->stat_scene->mesh_nrmls, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Scene Mesh Normal Buffer. %i\n", err);
        exit(1);
    }
    rctx->stat_scene->cl_mesh_index_buffer = clCreateBuffer(rctx->rcl->context,
                                                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                            sizeof(int)*rctx->stat_scene->num_mesh_indices*3,
                                                            rctx->stat_scene->mesh_indices, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Scene Mesh Index Buffer. %i\n", err);
        exit(1);
    }
}


void scene_resource_push(raytracer_context* rctx)
{
    int err;

    if(rctx->stat_scene->meshes_changed)
    {
        clEnqueueWriteBuffer (	rctx->rcl->commands,
                                rctx->stat_scene->cl_mesh_buffer, //TODO: make
                                CL_TRUE,
                                0,
                                sizeof(mesh)*rctx->stat_scene->num_meshes, //TODO: get from scene
                                rctx->stat_scene->meshes,
                                0,
                                NULL,
                                NULL);
    }

    if(rctx->stat_scene->spheres_changed)
    {
        clEnqueueWriteBuffer (	rctx->rcl->commands,
                                rctx->stat_scene->cl_sphere_buffer, //TODO: make
                                CL_TRUE,
                                0,
                                sizeof(sphere)*rctx->stat_scene->num_spheres, //TODO: get from scene
                                rctx->stat_scene->spheres,
                                0,
                                NULL,
                                NULL);
    }

    if(rctx->stat_scene->planes_changed)
    {
        clEnqueueWriteBuffer (	rctx->rcl->commands,
                                rctx->stat_scene->cl_plane_buffer, //TODO: make
                                CL_TRUE,
                                0,
                                sizeof(plane)*rctx->stat_scene->num_planes, //TODO: get from scene
                                rctx->stat_scene->planes,
                                0,
                                NULL,
                                NULL);
    }


    if(rctx->stat_scene->materials_changed)
    {
        clEnqueueWriteBuffer (	rctx->rcl->commands,
                                rctx->stat_scene->cl_material_buffer, //TODO: make
                                CL_TRUE,
                                0,
                                sizeof(material)*rctx->stat_scene->num_materials, //TODO: get from scene
                                rctx->stat_scene->materials,
                                0,
                                NULL,
                                NULL);
    }



}
