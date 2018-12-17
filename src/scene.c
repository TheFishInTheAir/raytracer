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
    ASRT_CL("Error Creating OpenCL Scene Sphere Buffer.");

    rctx->stat_scene->cl_plane_buffer = clCreateBuffer(rctx->rcl->context,
                                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                       sizeof(plane)*rctx->stat_scene->num_planes,
                                                       rctx->stat_scene->planes, &err);
    ASRT_CL("Error Creating OpenCL Scene Plane Buffer.");


    rctx->stat_scene->cl_material_buffer = clCreateBuffer(rctx->rcl->context,
                                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                          sizeof(material)*
                                                          rctx->stat_scene->num_materials,
                                                          rctx->stat_scene->materials, &err);
        ASRT_CL("Error Creating OpenCL Scene Plane Buffer.");


    //Mesh
    rctx->stat_scene->cl_mesh_buffer = clCreateBuffer(rctx->rcl->context,
                                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                      rctx->stat_scene->num_meshes==0 ? 1 :
                                                      sizeof(mesh)*rctx->stat_scene->num_meshes,
                                                      rctx->stat_scene->meshes, &err);
    ASRT_CL("Error Creating OpenCL Scene Mesh Buffer.");

    //mesh data is stored as images for faster access
    rctx->stat_scene->cl_mesh_vert_buffer =
        gen_1d_image(rctx, rctx->stat_scene->num_mesh_verts==0 ? 1 :
                     sizeof(vec3)*rctx->stat_scene->num_mesh_verts,
                     rctx->stat_scene->mesh_verts);

    rctx->stat_scene->cl_mesh_nrml_buffer =
        gen_1d_image(rctx, rctx->stat_scene->num_mesh_nrmls==0 ? 1 :
                     sizeof(vec3)*rctx->stat_scene->num_mesh_nrmls,
                     rctx->stat_scene->mesh_nrmls);

    rctx->stat_scene->cl_mesh_index_buffer =
        gen_1d_image(rctx, rctx->stat_scene->num_mesh_indices==0 ? 1 :
                     sizeof(int)*
                     rctx->stat_scene->num_mesh_indices,//maybe
                     rctx->stat_scene->mesh_indices);
}


void scene_resource_push(raytracer_context* rctx)
{
    int err;

    if(rctx->stat_scene->meshes_changed)
    {
        clEnqueueWriteBuffer (	rctx->rcl->commands,
                                rctx->stat_scene->cl_mesh_buffer,
                                CL_TRUE,
                                0,
                                sizeof(mesh)*rctx->stat_scene->num_meshes,
                                rctx->stat_scene->meshes,
                                0,
                                NULL,
                                NULL);
    }

    if(rctx->stat_scene->spheres_changed)
    {
        clEnqueueWriteBuffer (	rctx->rcl->commands,
                                rctx->stat_scene->cl_sphere_buffer,
                                CL_TRUE,
                                0,
                                sizeof(sphere)*rctx->stat_scene->num_spheres,
                                rctx->stat_scene->spheres,
                                0,
                                NULL,
                                NULL);
    }

    if(rctx->stat_scene->planes_changed)
    {
        clEnqueueWriteBuffer (	rctx->rcl->commands,
                                rctx->stat_scene->cl_plane_buffer,
                                CL_TRUE,
                                0,
                                sizeof(plane)*rctx->stat_scene->num_planes,
                                rctx->stat_scene->planes,
                                0,
                                NULL,
                                NULL);
    }


    if(rctx->stat_scene->materials_changed)
    {
        clEnqueueWriteBuffer (	rctx->rcl->commands,
                                rctx->stat_scene->cl_material_buffer,
                                CL_TRUE,
                                0,
                                sizeof(material)*rctx->stat_scene->num_materials,
                                rctx->stat_scene->materials,
                                0,
                                NULL,
                                NULL);
    }
}
