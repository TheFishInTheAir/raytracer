#include <path_raytracer.h>

path_raytracer_context* init_path_raytracer_context(struct _rt_ctx* rctx)
{
    path_raytracer_context* prctx = (path_raytracer_context*) malloc(sizeof(path_raytracer_context));
    prctx->rctx = rctx;
    prctx->up_to_date = false;
    prctx->num_samples = 128;//arbitrary default
    int err;
    printf("Generating Pathtracer Buffers...\n");
    prctx->cl_path_fresh_frame_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_WRITE,
                                                       rctx->width*rctx->height*sizeof(vec4),
                                                       NULL, &err);
    ASRT_CL("Error Creating OpenCL Fresh Frame Buffer.");
    prctx->cl_path_output_buffer = clCreateBuffer(rctx->rcl->context,
                                                  CL_MEM_READ_WRITE,
                                                  rctx->width*rctx->height*sizeof(vec4),
                                                  NULL, &err);
    ASRT_CL("Error Creating OpenCL Path Tracer Output Buffer.");

    printf("Generated Pathtracer Buffers...\n");
    return prctx;
}

//NOTE: the more divisions the slower.
#define WATCHDOG_DIVISIONS_X 2
#define WATCHDOG_DIVISIONS_Y 2
void path_raytracer_path_trace(path_raytracer_context* prctx)
{
    int err;

    const unsigned x_div = prctx->rctx->width/WATCHDOG_DIVISIONS_X;
    const unsigned y_div = prctx->rctx->height/WATCHDOG_DIVISIONS_Y;

    //scene_resource_push(rctx); //Update Scene buffers if necessary.

    cl_kernel kernel = prctx->rctx->program->raw_kernels[PATH_TRACE_KRNL_INDX]; //just use the first one

    float zeroed[] = {0., 0., 0., 1.};
    float* result = matvec_mul(prctx->rctx->stat_scene->camera_world_matrix, zeroed);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &prctx->cl_path_fresh_frame_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &prctx->rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &prctx->rctx->stat_scene->cl_material_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &prctx->rctx->stat_scene->cl_sphere_buffer);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &prctx->rctx->stat_scene->cl_plane_buffer);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &prctx->rctx->stat_scene->cl_mesh_buffer);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &prctx->rctx->stat_scene->cl_mesh_index_buffer.image);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &prctx->rctx->stat_scene->cl_mesh_vert_buffer.image);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &prctx->rctx->stat_scene->cl_mesh_nrml_buffer.image);

    clSetKernelArg(kernel, 9,  sizeof(int),     &prctx->rctx->width);
    clSetKernelArg(kernel, 10, sizeof(vec4),    result);
    clSetKernelArg(kernel, 11, sizeof(int),     &prctx->current_sample); //NOTE: I don't think this is used

    size_t global[2] = {x_div, y_div};

    //NOTE: tripping watchdog timer
    if(global[0]*WATCHDOG_DIVISIONS_X*global[1]*WATCHDOG_DIVISIONS_Y!=
       prctx->rctx->width*prctx->rctx->height)
    {
        printf("Watchdog divisions are incorrect!\n");
        exit(1);
    }

    size_t offset[2];

    for(int x = 0; x < WATCHDOG_DIVISIONS_X; x++)
    {
        for(int y = 0; y < WATCHDOG_DIVISIONS_Y; y++)
        {
            offset[0] = x_div*x;
            offset[1] = y_div*y;
            err = clEnqueueNDRangeKernel(prctx->rctx->rcl->commands, kernel, 2,
                                         offset, global, NULL, 0, NULL, NULL);
            ASRT_CL("Failed to execute path trace kernel");
        }
    }

    err = clFinish(prctx->rctx->rcl->commands);
    ASRT_CL("Something happened while executing path trace kernel");
}


void path_raytracer_average_buffers(path_raytracer_context* prctx)
{
    int err;

    cl_kernel kernel = prctx->rctx->program->raw_kernels[F_BUFFER_AVG_KRNL_INDX];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &prctx->cl_path_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &prctx->cl_path_fresh_frame_buffer);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &prctx->rctx->width);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &prctx->rctx->height);
    clSetKernelArg(kernel, 4, sizeof(unsigned int), &prctx->num_samples);
    clSetKernelArg(kernel, 5, sizeof(unsigned int), &prctx->current_sample);

    size_t global;
    size_t local = get_workgroup_size(prctx->rctx, kernel);

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global =  prctx->rctx->width*prctx->rctx->height;
    err = clEnqueueNDRangeKernel(prctx->rctx->rcl->commands, kernel, 1, NULL,
                                 &global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kernel");
    err = clFinish(prctx->rctx->rcl->commands);
    ASRT_CL("Something happened while waiting for kernel to finish");
}

void path_raytracer_push_path(path_raytracer_context* prctx)
{
    int err;

    cl_kernel kernel = prctx->rctx->program->raw_kernels[F_BUF_TO_BYTE_BUF_KRNL_INDX];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &prctx->rctx->cl_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &prctx->cl_path_output_buffer);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &prctx->rctx->width);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &prctx->rctx->height);



    size_t global;
    size_t local = get_workgroup_size(prctx->rctx, kernel);

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global =  prctx->rctx->width*prctx->rctx->height;
    err = clEnqueueNDRangeKernel(prctx->rctx->rcl->commands, kernel, 1,
                                 NULL, &global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kernel");

    err = clFinish(prctx->rctx->rcl->commands);
    ASRT_CL("Something happened while waiting for kernel to finish");


    err = clEnqueueReadBuffer(prctx->rctx->rcl->commands, prctx->rctx->cl_output_buffer, CL_TRUE, 0,
                              prctx->rctx->width*prctx->rctx->height*sizeof(int),
                              prctx->rctx->output_buffer,
                              0, NULL, NULL );
    ASRT_CL("Failed to read output array");
    //printf("RENDER\n");

}


void path_raytracer_render(path_raytracer_context* prctx)
{
    int local_start_time = os_get_time_mili(abst);
    prctx->current_sample++;
    if(prctx->current_sample>prctx->num_samples)
    {
        prctx->render_complete = true;
        printf("Render took %d ms\n", os_get_time_mili(abst)-prctx->start_time);
        return;
    }
    _raytracer_gen_ray_buffer(prctx->rctx);

    path_raytracer_path_trace(prctx);

    if(prctx->current_sample == 1) //needs to be here
    {
        int err;
        err = clEnqueueCopyBuffer (	prctx->rctx->rcl->commands,
                                    prctx->cl_path_fresh_frame_buffer,
                                    prctx->cl_path_output_buffer,
                                    0,
                                    0,
                                    prctx->rctx->width*prctx->rctx->height*sizeof(vec4),
                                    0,
                                    0,
                                    NULL);
        ASRT_CL("Error copying OpenCL Output Buffer");

        err = clFinish(prctx->rctx->rcl->commands);
        ASRT_CL("Something happened while waiting for copy to finish");
    }
    path_raytracer_average_buffers(prctx);
    path_raytracer_push_path(prctx);
    printf("Total time for sample group: %d\n", os_get_time_mili(abst)-local_start_time);
}

void path_raytracer_prepass(path_raytracer_context* prctx)
{
    raytracer_prepass(prctx->rctx); //Nothing Special
    prctx->current_sample = 0;
    prctx->start_time = os_get_time_mili(abst);
}
