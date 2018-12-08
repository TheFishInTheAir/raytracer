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
    ic_init(rctx);
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
                         all_kernels_cl, //NOTE: Binary resource
                         kernels, NUM_KERNELS, rctx->program,
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
    rctx->cl_path_output_buffer = clCreateBuffer(rctx->rcl->context,
                                         CL_MEM_READ_WRITE,
                                         rctx->width*rctx->height*sizeof(vec4),
                                         NULL, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Path output buffer Buffer. %i\n", err);
        exit(1);
    }
    rctx->cl_output_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_WRITE,
                                            rctx->width*rctx->height*4, NULL, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Output Buffer. %i\n", err);
        exit(1);
    }

    rctx->cl_path_fresh_frame_buffer = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_WRITE,
                                                 rctx->width*rctx->height*sizeof(vec4), NULL, &err);
    if(err!=CL_SUCCESS)
    {
        printf("Error Creating OpenCL Fresh Frame Buffer. %i\n", err);
        exit(1);
    }

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

#define JANK_SAMPLES 140
void raytracer_refined_render(raytracer_context* rctx)
{
    static unsigned int magic = 0;
    magic++;

    if(magic>JANK_SAMPLES)
        return;
    _raytracer_gen_ray_buffer(rctx);

    //ic_screenspace(rctx);

    //return;



    _raytracer_path_trace(rctx, magic);

    if(magic==1)
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
        if(err!=CL_SUCCESS)
        {
            printf("Error Copying OpenCL Output Buffer. %i\n", err);
            exit(1);
        }
        clFinish(rctx->rcl->commands);
    }

    _raytracer_average_buffers(rctx, magic);
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
    size_t local = 0;
    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    global =  rctx->width*rctx->height;
    err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %i\n",err);
        exit(1);
    }

    //Wait for completion
    clFinish(rctx->rcl->commands);

}
void _raytracer_average_buffers(raytracer_context* rctx, unsigned int sample_num)
{
    int err;
    int samples = JANK_SAMPLES;
    cl_kernel kernel = rctx->program->raw_kernels[F_BUFFER_AVG_KRNL_INDX];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_path_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rctx->cl_path_fresh_frame_buffer);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &rctx->width);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &rctx->height);
    clSetKernelArg(kernel, 4, sizeof(unsigned int), &samples);
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
                              rctx->width*rctx->height*sizeof(int), rctx->output_buffer, 0, NULL, NULL );
    ASRT_CL("Failed to read output array");

}


void _raytracer_path_trace(raytracer_context* rctx, unsigned int sample_num) //TODO: do more path tracing stuff here
{
    int err;

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
    clSetKernelArg(kernel, 10, sizeof(int),     &rctx->height);
    clSetKernelArg(kernel, 11, sizeof(vec4),    result);
    clSetKernelArg(kernel, 12, sizeof(int),     &sample_num); //NOTE: I don't think this is used

    //free(result);

    size_t global; //TODO: optimize
    size_t local = 0;
    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(local), &local, NULL); //NOTE: we don't need to do this every time
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
    size_t local = 0;
    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(local), &local, NULL); //NOTE: we don't need to do this every time
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
