#include <raytracer.h>
#include <parallel.h>

//binary resources
#include <test.cl.h> //test kernel



//NOTE: we are assuming the output buffer will be the right size
raytracer_context* raytracer_init(unsigned int width, unsigned int height,
                                      uint32_t* output_buffer, rcl_ctx* rcl)
{
    raytracer_context* rctx = (raytracer_context*) malloc(sizeof(raytracer_context));
    rctx->width  = width;
    rctx->height = height;
    rctx->ray_buffer = (float*) malloc(width * height * sizeof(float)*3);
    rctx->output_buffer = output_buffer;
    //rctx->fresh_buffer = (uint32_t*) malloc(width * height * sizeof(uint32_t));
    rctx->rcl = rcl;
    rctx->program = (rcl_program*) malloc(sizeof(rcl_program));
    rctx->ic_ctx = (ic_context*) malloc(sizeof(ic_context));
    //ic_init(rctx);
    rctx->render_complete = false;
    rctx->num_samples     = 64; //NOTE: arbitrary default
    rctx->current_sample  = 0;

    return rctx;
}

void raytracer_cl_prepass(raytracer_context* rctx)
{
    //CL init
    printf("Building Scene Kernels...\n");

    int err = CL_SUCCESS;

    //Kernels
    char* kernels[] = KERNELS;

    //Macros
    //char os_macro[64];
    #ifdef _WIN32
    char os_macro[] = "#define _WIN32 1";
    #else
    char os_macro[] = "#define _OSX 1";
    #endif
    char sphere_macro[64];
    sprintf(sphere_macro, "#define SCENE_NUM_SPHERES %i", rctx->stat_scene->num_spheres);
    char plane_macro[64];
    sprintf(plane_macro, "#define SCENE_NUM_PLANES %i", rctx->stat_scene->num_planes);
    char index_macro[64];
    sprintf(index_macro, "#define SCENE_NUM_INDICES %i", rctx->stat_scene->num_mesh_indices);
    char mesh_macro[64];
    sprintf(mesh_macro, "#define SCENE_NUM_MESHES %i", rctx->stat_scene->num_meshes);
    char material_macro[64];
    sprintf(material_macro, "#define SCENE_NUM_MATERIALS %i", rctx->stat_scene->num_materials);
    char* macros[]  = {sphere_macro, plane_macro, mesh_macro, index_macro,
                       material_macro, os_macro};

    {

        load_program_raw(rctx->rcl,
                         all_kernels_cl, //NOTE: Binary resource
                         kernels, NUM_KERNELS, rctx->program,
                         macros, 6);
    }
    //Buffers
    rctx->cl_ray_buffer = clCreateBuffer(rctx->rcl->context,
                                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         rctx->width*rctx->height*sizeof(float)*3,
                                         rctx->ray_buffer, &err);
    ASRT_CL("Error Creating OpenCL Ray Buffer.");
    rctx->cl_path_output_buffer = clCreateBuffer(rctx->rcl->context,
                                         CL_MEM_READ_WRITE,
                                         rctx->width*rctx->height*sizeof(vec4),
                                         NULL, &err);
    ASRT_CL("Error Creating OpenCL Path Tracer Output Buffer.");

    rctx->cl_output_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_WRITE,
                                            rctx->width*rctx->height*4, NULL, &err);
    ASRT_CL("Error Creating OpenCL Output Buffer.");

    //TODO: all output buffers and frame buffers should be images.
    rctx->cl_path_fresh_frame_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_WRITE,
                                                 rctx->width*rctx->height*sizeof(vec4), NULL, &err);
    ASRT_CL("Error Creating OpenCL Fresh Frame Buffer.");

	printf("Pushing Scene Resources.\n");
	scene_init_resources(rctx);

    printf("Built Scene Kernels.\n");
}

void raytracer_prepass(raytracer_context* rctx)
{
    printf("Starting Raytracer Prepass.\n");


    raytracer_cl_prepass(rctx);


    printf("Finished Raytracer Prepass.\n");

} //TODO: implement
void raytracer_render(raytracer_context* rctx)
{
    _raytracer_gen_ray_buffer(rctx);

    _raytracer_cast_rays(rctx);
}

//#define JANK_SAMPLES 32
void raytracer_refined_render(raytracer_context* rctx)
{
    rctx->current_sample++;
    if(rctx->current_sample>rctx->num_samples)
    {
        rctx->render_complete = true;
        return;
    }
    _raytracer_gen_ray_buffer(rctx);

    _raytracer_path_trace(rctx, rctx->current_sample);

    if(rctx->current_sample==1) //really terrible place for path tracer initialization...
    {
        int err;
        char pattern = 0;
        err = clEnqueueCopyBuffer (	rctx->rcl->commands,
                                    rctx->cl_path_fresh_frame_buffer,
                                    rctx->cl_path_output_buffer,
                                    0,
                                    0,
                                    rctx->width*rctx->height*sizeof(vec4),
                                    0,
                                    0,
                                    NULL);
        ASRT_CL("Error copying OpenCL Output Buffer");

        err = clFinish(rctx->rcl->commands);
        ASRT_CL("Something happened while waiting for copy to finish");
    }

    _raytracer_average_buffers(rctx, rctx->current_sample);
    _raytracer_push_path(rctx);

}

void _raytracer_gen_ray_buffer(raytracer_context* rctx)
{
    int err;

    cl_kernel kernel = rctx->program->raw_kernels[RAY_BUFFER_KRNL_INDX];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 1, sizeof(unsigned int), &rctx->width);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &rctx->height);
    clSetKernelArg(kernel, 3, sizeof(mat4), rctx->stat_scene->camera_world_matrix);


    size_t global;


    global =  rctx->width*rctx->height;
    err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kernel");


    //Wait for completion
    err = clFinish(rctx->rcl->commands);
    ASRT_CL("Something happened while waiting for kernel raybuf to finish");


}
void _raytracer_average_buffers(raytracer_context* rctx, unsigned int sample_num)
{
    int err;

    cl_kernel kernel = rctx->program->raw_kernels[F_BUFFER_AVG_KRNL_INDX];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_path_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rctx->cl_path_fresh_frame_buffer);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &rctx->width);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &rctx->height);
    clSetKernelArg(kernel, 4, sizeof(unsigned int), &rctx->num_samples);
    clSetKernelArg(kernel, 5, sizeof(unsigned int), &sample_num);

    size_t global;
    size_t local = get_workgroup_size(rctx, kernel);

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global =  rctx->width*rctx->height;
    err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kernel")
    err = clFinish(rctx->rcl->commands);
    ASRT_CL("Something happened while waiting for kernel to finish");



}

void _raytracer_push_path(raytracer_context* rctx)
{
    int err;

    cl_kernel kernel = rctx->program->raw_kernels[F_BUF_TO_BYTE_BUF_KRNL_INDX];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rctx->cl_path_output_buffer);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &rctx->width);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &rctx->height);



    size_t global;
    size_t local = get_workgroup_size(rctx, kernel);

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global =  rctx->width*rctx->height;
    err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kernel");


    err = clFinish(rctx->rcl->commands);
    ASRT_CL("Something happened while waiting for kernel to finish");

    err = clEnqueueReadBuffer(rctx->rcl->commands, rctx->cl_output_buffer, CL_TRUE, 0,
                              rctx->width*rctx->height*sizeof(int), rctx->output_buffer,
                              0, NULL, NULL );
    ASRT_CL("Failed to read output array");

}

//NOTE: the more divisions the slower.
#define WATCHDOG_DIVISIONS_X 2
#define WATCHDOG_DIVISIONS_Y 2
void _raytracer_path_trace(raytracer_context* rctx, unsigned int sample_num)
{
    int err;

    const unsigned x_div = rctx->width/WATCHDOG_DIVISIONS_X;
    const unsigned y_div = rctx->height/WATCHDOG_DIVISIONS_Y;

    //scene_resource_push(rctx); //Update Scene buffers if necessary.

    cl_kernel kernel = rctx->program->raw_kernels[PATH_TRACE_KRNL_INDX]; //just use the first one

    float zeroed[] = {0., 0., 0., 1.};
    float* result = matvec_mul(rctx->stat_scene->camera_world_matrix, zeroed);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_path_fresh_frame_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &rctx->stat_scene->cl_material_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &rctx->stat_scene->cl_sphere_buffer);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &rctx->stat_scene->cl_plane_buffer);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_buffer);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_index_buffer);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_vert_buffer);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_nrml_buffer);

    clSetKernelArg(kernel, 9,  sizeof(int),     &rctx->width);
    clSetKernelArg(kernel, 10, sizeof(vec4),    result);
    clSetKernelArg(kernel, 11, sizeof(int),     &sample_num); //NOTE: I don't think this is used

    size_t global[2] = {x_div, y_div};

    //NOTE: tripping watchdog timer
    if(global[0]*WATCHDOG_DIVISIONS_X*global[1]*WATCHDOG_DIVISIONS_Y!=rctx->width*rctx->height)
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
            err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 2,
                                         offset, global, NULL, 0, NULL, NULL);
            ASRT_CL("Failed to execute path trace kernel");
        }
    }

    err = clFinish(rctx->rcl->commands);
    ASRT_CL("Something happened while executing path trace kernel");
}


void _raytracer_cast_rays(raytracer_context* rctx) //TODO: do more path tracing stuff here
{
    int err;



    scene_resource_push(rctx); //Update Scene buffers if necessary.


    cl_kernel kernel = rctx->program->raw_kernels[RAY_CAST_KRNL_INDX]; //just use the first one

    float zeroed[] = {0., 0., 0., 1.};
    float* result = matvec_mul(rctx->stat_scene->camera_world_matrix, zeroed);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &rctx->stat_scene->cl_material_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &rctx->stat_scene->cl_sphere_buffer);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &rctx->stat_scene->cl_plane_buffer);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_buffer);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_index_buffer);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_vert_buffer);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_nrml_buffer);

    clSetKernelArg(kernel, 9, sizeof(unsigned int), &rctx->width);
    clSetKernelArg(kernel, 10, sizeof(unsigned int), &rctx->height);
    clSetKernelArg(kernel, 11, sizeof(float)*4, result); //we only need 3
    //free(result);

    size_t global;

    global =  rctx->width*rctx->height;
    err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to Execute Kernel");

    err = clFinish(rctx->rcl->commands);
    ASRT_CL("Something happened during kernel execution");

    err = clEnqueueReadBuffer(rctx->rcl->commands, rctx->cl_output_buffer, CL_TRUE, 0,
                              rctx->width*rctx->height*sizeof(int), rctx->output_buffer, 0, NULL, NULL );
    ASRT_CL("Failed to read output array");

}
