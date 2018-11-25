#include <raytracer.h>
#include <parallel.h>
//NOTE: we are assuming the output buffer will be the right size
raytracer_context* raytracer_init(unsigned int width, unsigned int height,
                                      uint32_t* output_buffer, rcl_ctx* rcl)
{
    raytracer_context* rctx = (raytracer_context*) malloc(sizeof(raytracer_context));
    rctx->width  = width;
    rctx->height = height;
    rctx->ray_buffer = (float*) malloc(width * height * sizeof(float)*3);
    rctx->output_buffer = output_buffer;
    rctx->rcl = rcl;
    rctx->program = (rcl_program*) malloc(sizeof(rcl_program));

    return rctx;
}

void raytracer_cl_prepass(raytracer_context* rctx)
{
    //CL init
    printf("Building Scene Kernels...\n");

    int err = CL_SUCCESS;

    //Kernels
    char* kernels[] = {"cast_ray_test", "generate_rays"};
    rctx->ray_cast_kernel_index   = 0;
    rctx->ray_buffer_kernel_index = 1;
    printf("test1\n");
    //Macros
    char sphere_macro[64];
    sprintf(sphere_macro, "#define SCENE_NUM_SPHERES %i", rctx->stat_scene->num_spheres);
    char plane_macro[64];
    sprintf(plane_macro, "#define SCENE_NUM_PLANES %i", rctx->stat_scene->num_planes);
    char* macros[]  = {sphere_macro, plane_macro};
    printf("test2\n");

    //NOTE: Temp hardcoded URL
    load_program_url(rctx->rcl,
                     "C:\\Users\\Ethan Breit\\AppData\\Roaming\\Emacs\\Western\\10\\Science\\Raytracer\\src\\kernels\\test.cl",
                     kernels, 2, rctx->program,
                     macros, 2);
    printf("test2.5\n");

    printf("test3\n");

    //Buffers
    rctx->cl_ray_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         rctx->width*rctx->height*sizeof(float)*3, rctx->ray_buffer, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Ray Buffer.\n");
        exit(1);
    }
    rctx->cl_output_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_WRITE_ONLY,
                                            rctx->width*rctx->height*4, NULL, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Output Buffer.\n");
        exit(1);
    }
    //Scene Buffers
    rctx->cl_sphere_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(float)*4*rctx->stat_scene->num_spheres,
                                            rctx->stat_scene->spheres, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Scene Sphere Buffer.\n");
        exit(1);
    }

    rctx->cl_plane_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sizeof(float)*6*rctx->stat_scene->num_planes,
                                            rctx->stat_scene->planes, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Scene Plane Buffer.\n");
        exit(1);
    }
    //TODO: add sphere buffer (also buffers for the rest of the primitives)
    printf("Built Scene Kernels.\n");
}

//NOTE: we need a scene by this point.
void raytracer_prepass(raytracer_context* rctx)
{
    printf("Starting Raytracer Prepass.\n");


    raytracer_cl_prepass(rctx);


    printf("Finished Raytracer Prepass.\n");

} //TODO: implement
void raytracer_render(raytracer_context* rctx) //TODO: implement
{
    _raytracer_gen_ray_buffer(rctx);

    _raytracer_cast_rays(rctx);
}

void _raytracer_gen_ray_buffer(raytracer_context* rctx)
{
    int err;

    cl_kernel kernel = rctx->program->raw_kernels[rctx->ray_buffer_kernel_index]; //just use the first one

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 1, sizeof(unsigned int), &rctx->width);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &rctx->height);


    size_t global;
    size_t local = 0;

    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(local), &local, NULL); //NOTE: we don't need to do this every time
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // TODO: optimize work groups.
    global =  rctx->width*rctx->height;
    err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel! %i\n",err);
        exit(1);
    }

    //Wait for completion
    clFinish(rctx->rcl->commands);

    //Read Data from opencl ray buffer to our ray buffer
    //r = clEnqueueReadBuffer(rctx->rcl->commands, rctx->cl_ray_buffer, CL_TRUE, 0,
    //                        rctx->width*rctx->height*sizeof(float)*3, rctx->ray_buffer, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
}

void _raytracer_cast_rays(raytracer_context* rctx) //TODO: do more path tracing stuff here
{
    int err;



    clEnqueueWriteBuffer (	rctx->rcl->commands,
                            rctx->cl_sphere_buffer, //TODO: make
                            CL_TRUE,
                            0,
                            sizeof(float)*4*rctx->stat_scene->num_spheres, //TODO: get from scene
                            rctx->stat_scene->spheres,
                            0,
                            NULL,
                            NULL);
    clEnqueueWriteBuffer (	rctx->rcl->commands,
                            rctx->cl_plane_buffer, //TODO: make
                            CL_TRUE,
                            0,
                            sizeof(float)*6*rctx->stat_scene->num_planes, //TODO: get from scene
                            rctx->stat_scene->planes,
                            0,
                            NULL,
                            NULL);

/*  clEnqueueWriteBuffer (	rctx->rcl->commands,
                            rctx->cl_ray_buffer, //TODO: make
                            CL_TRUE,
                            0,
                            rctx->width*rctx->height*sizeof(float)*3, //TODO: get from scene
                            rctx->ray_buffer,
                            0,
                            NULL,
                            NULL);*/



    cl_kernel kernel = rctx->program->raw_kernels[rctx->ray_cast_kernel_index]; //just use the first one

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &rctx->cl_sphere_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &rctx->cl_plane_buffer);
    clSetKernelArg(kernel, 4, sizeof(unsigned int), &rctx->width);
    clSetKernelArg(kernel, 5, sizeof(unsigned int), &rctx->height);


    size_t global;
    size_t local = 0;

    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return;
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    //printf("STARTING\n");
    global =  rctx->width*rctx->height;
    err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel! %i\n",err);
        return;
    }


    clFinish(rctx->rcl->commands);

    //printf("STOPPING\n");

    err = clEnqueueReadBuffer(rctx->rcl->commands, rctx->cl_output_buffer, CL_TRUE, 0,
                              rctx->width*rctx->height*sizeof(int), rctx->output_buffer, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
}
