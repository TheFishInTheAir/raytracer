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
    rctx->ray_buffer = (float*) malloc(width * height * sizeof(ray));
    rctx->output_buffer = output_buffer;
    //rctx->fresh_buffer = (uint32_t*) malloc(width * height * sizeof(uint32_t));
    rctx->rcl = rcl;
    rctx->program = (rcl_program*) malloc(sizeof(rcl_program));
    rctx->ic_ctx = (ic_context*) malloc(sizeof(ic_context));
    //ic_init(rctx);
    rctx->render_complete = false;
    rctx->num_samples     = 64; //NOTE: arbitrary default
    rctx->current_sample  = 0;
    rctx->event_position = 0;
    rctx->block_size_y = 0;
    rctx->block_size_x = 0;
    return rctx;
}

void raytracer_build_kernels(raytracer_context* rctx)
{
    printf("Building Kernels...\n");
    char* kernels[] = KERNELS;
    printf("Generating Kernel Macros...\n");
    //Macros
    unsigned int num_macros = 0;
    #ifdef _WIN32
    char os_macro[] = "#define _WIN32 1";
    #else
    char os_macro[] = "#define _OSX 1";
    #endif
    num_macros++;

    MACRO_GEN(sphere_macro,   SCENE_NUM_SPHERES %i, rctx->stat_scene->num_spheres, num_macros);
    MACRO_GEN(plane_macro,    SCENE_NUM_PLANES  %i, rctx->stat_scene->num_planes,  num_macros);
    MACRO_GEN(index_macro,    SCENE_NUM_INDICES %i, rctx->stat_scene->num_mesh_indices, num_macros);
    MACRO_GEN(mesh_macro,     SCENE_NUM_MESHES  %i, rctx->stat_scene->num_meshes, num_macros);
    MACRO_GEN(material_macro, SCENE_NUM_MATERIALS  %i, rctx->stat_scene->num_materials, num_macros);
    MACRO_GEN(blockx_macro,   BLOCKSIZE_X  %i, rctx->block_size_x, num_macros);
    MACRO_GEN(blocky_macro,   BLOCKSIZE_Y  %i, rctx->block_size_y, num_macros);

    char min_macro[64];
    sprintf(min_macro, "#define SCENE_MIN (%f, %f, %f)",
            rctx->stat_scene->kdt->bounds.min[0],
            rctx->stat_scene->kdt->bounds.min[1],
            rctx->stat_scene->kdt->bounds.min[2]);
    num_macros++;
    char max_macro[64];
    sprintf(max_macro, "#define SCENE_MAX (%f, %f, %f)",
            rctx->stat_scene->kdt->bounds.max[0],
            rctx->stat_scene->kdt->bounds.max[1],
            rctx->stat_scene->kdt->bounds.max[2]);
    num_macros++;


    //TODO: do something better than this
    char* macros[]  = {sphere_macro, plane_macro, mesh_macro, index_macro,
                       material_macro, os_macro, blockx_macro, blocky_macro,
                       min_macro, max_macro};
    printf("Macros Generated.\n");

    load_program_raw(rctx->rcl,
                     all_kernels_cl, //NOTE: Binary resource
                     kernels, NUM_KERNELS, rctx->program,
                     macros, num_macros);
    printf("Kernels built.\n");

}

void raytracer_build(raytracer_context* rctx)
{
    //CL init
    printf("Building Scene...\n");

    int err = CL_SUCCESS;

	printf("Initializing Scene Resources On GPU.\n");
	scene_init_resources(rctx);
    rctx->stat_scene->kdt->s = rctx->stat_scene;
	printf("Initialized Scene Resources On GPU.\n");


    printf("Building/Rebuilding k-d tree.\n");
    kd_tree_construct(rctx->stat_scene->kdt);
    printf("Done Building/Rebuilding k-d tree.\n");



    //Kernels
    raytracer_build_kernels(rctx);

    //Buffers
    printf("Generating Buffers...\n");
    rctx->cl_ray_buffer = clCreateBuffer(rctx->rcl->context,
                                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         rctx->width*rctx->height*sizeof(ray),
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

    printf("Generated Buffers...\n");
}

void raytracer_prepass(raytracer_context* rctx)
{
    printf("Starting Raytracer Prepass.\n");

    printf("Pushing Scene Resources.\n");
    scene_resource_push(rctx);
    printf("Finished Pushing Scene Resources.\n");

    printf("Finished Raytracer Prepass.\n");
}

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

    //Nothings wrong I just am currently refactoring this
    //_raytracer_average_buffers(rctx, rctx->current_sample);
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
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_index_buffer.image);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_vert_buffer.image);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_nrml_buffer.image);

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
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_index_buffer.image);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_vert_buffer.image);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_nrml_buffer.image);

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
