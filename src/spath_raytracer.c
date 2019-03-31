#include <spath_raytracer.h>
#include <kdtree.h>
#include <raytracer.h>

spath_raytracer_context* init_spath_raytracer_context(struct _rt_ctx* rctx)
{
    spath_raytracer_context* sprctx = (spath_raytracer_context*) malloc(sizeof(spath_raytracer_context));
    sprctx->rctx = rctx;
    sprctx->up_to_date = false;
    sprctx->num_samples = 32;//arbitrary default
    int err;
    printf("Generating Split Pathtracer Buffers...\n");
    sprctx->cl_path_fresh_frame_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_WRITE,
                                                       rctx->width*rctx->height*sizeof(vec4),
                                                       NULL, &err);
    ASRT_CL("Error Creating OpenCL Split Fresh Frame Buffer.");
    sprctx->cl_path_output_buffer = clCreateBuffer(rctx->rcl->context,
                                                  CL_MEM_READ_WRITE,
                                                  rctx->width*rctx->height*sizeof(vec4),
                                                  NULL, &err);
    ASRT_CL("Error Creating OpenCL Split Path Tracer Output Buffer.");

    sprctx->cl_path_collision_result_buffer = clCreateBuffer(rctx->rcl->context,
                                                             CL_MEM_READ_WRITE,
                                                             rctx->width*rctx->height*
                                                             sizeof(kd_tree_collision_result),
                                                             NULL, &err);
    ASRT_CL("Error Creating OpenCL Split Path Tracer Collision Result Buffer.");

    {
        unsigned int bad_buf[4*4+1];
        bad_buf[4*4] = 0;
        {
            //good thing this is the same transposed. Also this is stupid, but endorced by AMD
            unsigned int mat[4*4] = {0xffffffff, 0,          0,          0,
                                     0,          0xffffffff, 0,          0,
                                     0,          0,          0xffffffff, 0,
                                     0,          0,          0,          0xffffffff};
            memcpy(bad_buf, mat, 4*4*sizeof(unsigned int));
        }

        sprctx->cl_bad_api_design_buffer = clCreateBuffer(rctx->rcl->context,
                                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                          (4*4+1)*sizeof(float),
                                                          bad_buf, &err);
        ASRT_CL("Error Creating OpenCL BAD API DESIGN! Buffer.");

        err = clFinish(rctx->rcl->commands);
        ASRT_CL("Something happened while waiting for copy to finish");
    }
    printf("Generated Split Pathtracer Buffers.\n");
    return sprctx;
}

//NOTE: might need to do watchdog division for this, hopefully not though.
void spath_raytracer_kd_collision(spath_raytracer_context* sprctx)
{
    int err;

    cl_kernel kernel = sprctx->rctx->program->raw_kernels[KDTREE_INTERSECTION_INDX];

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sprctx->cl_path_collision_result_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &sprctx->rctx->cl_ray_buffer);

    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sprctx->cl_bad_api_design_buffer); //BAD

    clSetKernelArg(kernel, 3, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_buffer);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_index_buffer.image);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_vert_buffer.image);
    //clSetKernelArg(kernel, 6, sizeof(cl_mem), &sprctx->cl_bad_api_design_buffer); //TEST
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &sprctx->rctx->stat_scene->kdt->cl_kd_tree_buffer);
    //NOTE: WILL NOT WORK WITH ALL SITUATIONS:
    unsigned int num_rays = sprctx->rctx->width*sprctx->rctx->height;
    clSetKernelArg(kernel, 7, sizeof(unsigned int), &num_rays);


    //clGetDeviceInfo(prctx->rctx->CL_DEVICE_MAX_COMPUTE_UNITS)

    size_t global[1] = {1280};//sprctx->rctx->rcl->num_cores;
    size_t local[1]  = {32 * 4};//sprctx->rctx->rcl->simt_size; sprctx->rctx->rcl->num_simt_per_multiprocessor
    //printf("\n\n\n\n STARTING KD TREE INTERSECTION KERNEL \n\n\n\n\n");
    fflush(stdout);
    err = clEnqueueNDRangeKernel(sprctx->rctx->rcl->commands, kernel, 1,
                                 NULL, global, local, 0, NULL, NULL);
    ASRT_CL("Failed to execute kd tree traversal kernel");

    err = clFinish(sprctx->rctx->rcl->commands);
    ASRT_CL("Something happened while executing kd tree traversal kernel");
    printf("FINISHED KD TREE COLLISION\n");
}

void spath_raytracer_kd_test(spath_raytracer_context* sprctx)
{
    int err;

    cl_kernel kernel = sprctx->rctx->program->raw_kernels[KDTREE_TEST_DRAW_INDX];

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sprctx->rctx->cl_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &sprctx->cl_path_collision_result_buffer);

    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_material_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_buffer);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_index_buffer.image);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_vert_buffer.image);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_nrml_buffer.image);

    //clSetKernelArg(kernel, 6, sizeof(cl_mem), &sprctx->cl_bad_api_design_buffer); //TEST
    clSetKernelArg(kernel, 7, sizeof(unsigned int), &sprctx->rctx->width);
    //NOTE: WILL NOT WORK WITH ALL SITUATIONS:
    unsigned int num_rays = sprctx->rctx->width*sprctx->rctx->height;


    size_t global[1] = {num_rays};

    err = clEnqueueNDRangeKernel(sprctx->rctx->rcl->commands, kernel, 1,
                                 NULL, global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kd tree traversal kernel");

    err = clFinish(sprctx->rctx->rcl->commands);
    ASRT_CL("Something happened while executing kd tree traversal kernel");

    err = clEnqueueReadBuffer(sprctx->rctx->rcl->commands, sprctx->rctx->cl_output_buffer, CL_TRUE, 0,
                              sprctx->rctx->width*sprctx->rctx->height*sizeof(int),
                              sprctx->rctx->output_buffer, 0, NULL, NULL );
    ASRT_CL("Failed to read output array");
    //printf("FINISHED KD TREE COLLISION\n");
}

void spath_raytracer_render(spath_raytracer_context* sprctx)
{
    sprctx->current_sample++;
    if(sprctx->current_sample>sprctx->num_samples)
    {
        sprctx->render_complete = true;
        return;
    }
    spath_raytracer_kd_collision(sprctx);
    spath_raytracer_kd_test(sprctx);
}

void spath_raytracer_prepass(spath_raytracer_context* sprctx)
{
    raytracer_prepass(sprctx->rctx);
    sprctx->current_sample = 0;
    _raytracer_gen_ray_buffer(sprctx->rctx);
    printf("Finished Split Path Raytracer Prepass. \n");
}
