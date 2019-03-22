#include <ss_raytracer.h>
#include <scene.h>
#include <kdtree.h>
#include <raytracer.h>

//Single sweep, as close to real time as this thing can support.
void ss_raytracer_render(ss_raytracer_context* srctx)
{
    int err;

    //TODO: @REFACTOR and remove prefix underscore
    _raytracer_gen_ray_buffer(srctx->rctx);


    scene_resource_push(srctx->rctx); //Update Scene buffers if necessary.


    cl_kernel kernel = srctx->rctx->program->raw_kernels[RAY_CAST_KRNL_INDX]; //just use the first one

    float zeroed[] = {0., 0., 0., 1.};
    float* result = matvec_mul(srctx->rctx->stat_scene->camera_world_matrix, zeroed);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &srctx->rctx->cl_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &srctx->rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &srctx->rctx->stat_scene->cl_material_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &srctx->rctx->stat_scene->cl_sphere_buffer);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &srctx->rctx->stat_scene->cl_plane_buffer);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &srctx->rctx->stat_scene->cl_mesh_buffer);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &srctx->rctx->stat_scene->cl_mesh_index_buffer);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &srctx->rctx->stat_scene->cl_mesh_vert_buffer);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &srctx->rctx->stat_scene->cl_mesh_nrml_buffer);

    clSetKernelArg(kernel, 9, sizeof(unsigned int), &srctx->rctx->width);
    clSetKernelArg(kernel, 10, sizeof(unsigned int), &srctx->rctx->height);
    clSetKernelArg(kernel, 11, sizeof(float)*4, result); //we only need 3
    //free(result);

    size_t global;

    global =  srctx->rctx->width*srctx->rctx->height;
    err = clEnqueueNDRangeKernel(srctx->rctx->rcl->commands, kernel, 1, NULL, &global,
                                 NULL, 0, NULL, NULL);
    ASRT_CL("Failed to Execute Kernel");

    err = clFinish(srctx->rctx->rcl->commands);
    ASRT_CL("Something happened during kernel execution");

    err = clEnqueueReadBuffer(srctx->rctx->rcl->commands, srctx->rctx->cl_output_buffer, CL_TRUE, 0,
                              srctx->rctx->width*srctx->rctx->height*sizeof(int),
                              srctx->rctx->output_buffer, 0, NULL, NULL );
    ASRT_CL("Failed to read output array");

}
