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

    return rctx;
}

void raytracer_cl_prepass(raytracer_context* rctx)
{
    //CL init
    printf("Building Scene Kernels...\n");

    int err = CL_SUCCESS;

    //Kernels
    char* kernels[] = {"cast_ray_test", "generate_rays", "path_trace", "buffer_average"};
    rctx->ray_cast_kernel_index   = 0;
    rctx->ray_buffer_kernel_index = 1;
    rctx->path_trace_kernel_index = 2;
    rctx->buffer_average_kernel_index = 3;

    //Macros
    char sphere_macro[64];
    sprintf(sphere_macro, "#define SCENE_NUM_SPHERES %i", rctx->stat_scene->num_spheres);
    char plane_macro[64];
    sprintf(plane_macro, "#define SCENE_NUM_PLANES %i", rctx->stat_scene->num_planes);
    char mesh_macro[64];
    sprintf(mesh_macro, "#define SCENE_NUM_MESHES %i", rctx->stat_scene->num_meshes);
    char material_macro[64];
    sprintf(material_macro, "#define SCENE_NUM_MATERIALS %i", rctx->stat_scene->num_materials);
    char* macros[]  = {sphere_macro, plane_macro, mesh_macro, material_macro};

    {

        load_program_raw(rctx->rcl,
                         ___src_kernels_test_cl, //NOTE: Binary resource
                         kernels, 4, rctx->program,
                         macros, 4);
    }
    //Buffers
    rctx->cl_ray_buffer = clCreateBuffer(rctx->rcl->context,
                                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         rctx->width*rctx->height*sizeof(float)*3,
                                         rctx->ray_buffer, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Ray Buffer. %i\n", err);
        exit(1);
    }
    rctx->cl_output_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_WRITE,
                                            rctx->width*rctx->height*4, NULL, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Output Buffer. %i\n", err);
        exit(1);
    }

    char pattern = 0;
    err =  clEnqueueFillBuffer (rctx->rcl->commands,
                                rctx->cl_output_buffer,
                                &pattern, 1 ,0,
                                rctx->width*rctx->height*4,
                                0, NULL, NULL);
    if(err!=CL_SUCCESS)
    {
        printf("Error Zeroeing OpenCL Output Buffer. %i\n", err);
        exit(1);
    }
    rctx->cl_fresh_frame_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_WRITE,
                                            rctx->width*rctx->height*4, NULL, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Fresh Frame Buffer. %i\n", err);
        exit(1);
    }

	printf("Pushing Scene Resources.\n");
	scene_init_resources(rctx);

    //TODO: add sphere buffer (also buffers for the rest of the primitives)
    printf("Built Scene Kernels.\n");

    //clFinish(rctx->rcl->commands); //Shouldn't be necessary
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

void raytracer_refined_render(raytracer_context* rctx) //TODO: implement
{
    static unsigned int magic = 0;
    magic++;

    if(magic>1)
        return;

    _raytracer_gen_ray_buffer(rctx);

    _raytracer_path_trace(rctx, magic);

    if(magic==1)
    {
        int err;
        char pattern = 0;
        err = clEnqueueCopyBuffer (	rctx->rcl->commands,
                                    rctx->cl_fresh_frame_buffer,
                                    rctx->cl_output_buffer,
                                    0,
                                    0,
                                    rctx->width*rctx->height*sizeof(uint32_t),
                                    0,
                                    0,
                                    NULL);
        if(err!=CL_SUCCESS)
        {
            printf("Error Zeroeing OpenCL Output Buffer. %i\n", err);
            exit(1);
        }
        clFinish(rctx->rcl->commands);
    }

    _raytracer_average_buffers(rctx, magic);
}

void _raytracer_gen_ray_buffer(raytracer_context* rctx)
{
    int err;

    cl_kernel kernel = rctx->program->raw_kernels[rctx->ray_buffer_kernel_index]; //just use the first one

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 1, sizeof(unsigned int), &rctx->width);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &rctx->height);
    clSetKernelArg(kernel, 3, sizeof(mat4), rctx->stat_scene->camera_world_matrix);


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
    if (err != CL_SUCCESS)
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
void _raytracer_average_buffers(raytracer_context* rctx, int sample_num)
{
    int err;

    cl_kernel kernel = rctx->program->raw_kernels[rctx->buffer_average_kernel_index]; //just use the first one
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rctx->cl_fresh_frame_buffer);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &rctx->width);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &rctx->height);
    clSetKernelArg(kernel, 4, sizeof(unsigned int), &sample_num);


    size_t global; //TODO: optimize
    size_t local = 0;

    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
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
        exit(1);
    }


    clFinish(rctx->rcl->commands);

    err = clEnqueueReadBuffer(rctx->rcl->commands, rctx->cl_output_buffer, CL_TRUE, 0,
                              rctx->width*rctx->height*sizeof(int), rctx->output_buffer, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
}


void _raytracer_path_trace(raytracer_context* rctx, unsigned int sample_num) //TODO: do more path tracing stuff here
{
    int err;

    //scene_resource_push(rctx); //Update Scene buffers if necessary.

    cl_kernel kernel = rctx->program->raw_kernels[rctx->path_trace_kernel_index]; //just use the first one

    float zeroed[] = {0., 0., 0., 1.};
    float* result = matvec_mul(rctx->stat_scene->camera_world_matrix, zeroed);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_fresh_frame_buffer);
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
    clSetKernelArg(kernel, 12, sizeof(int), &sample_num); //we only need 3

    //free(result);

    size_t global; //TODO: optimize
    size_t local = 0;

    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
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
        exit(1);
    }


    clFinish(rctx->rcl->commands);

    //printf("STOPPING\n");


}


void _raytracer_cast_rays(raytracer_context* rctx) //TODO: do more path tracing stuff here
{
    int err;



    scene_resource_push(rctx); //Update Scene buffers if necessary.


    cl_kernel kernel = rctx->program->raw_kernels[rctx->ray_cast_kernel_index]; //just use the first one

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
    size_t local = 0;

    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
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
        exit(1);
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
