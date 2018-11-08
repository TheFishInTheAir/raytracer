#include <CL/opencl.h>

void cl_info()
{

    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (i = 0; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j+1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j+1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

        }

        free(devices);

    }

    free(platforms);
    return;
}

void create_context(rcl_ctx* ctx)
{
    int err = CL_SUCCESS;

    /* Identify a platform */
    err = clGetPlatformIDs(1, &ctx->platform_id, NULL);
    if(err != CL_SUCCESS)
    {
        printf("Couldn't identify a platform");
        return;
    }


    // Connect to a compute device
    //
    err = clGetDeviceIDs(ctx->platform_id, CL_DEVICE_TYPE_GPU, 1, &ctx->device_id, NULL);
    if (err != CL_SUCCESS)
    {
        if(err == CL_DEVICE_NOT_FOUND) {
            // CPU
            err = clGetDeviceIDs(ctx->platform_id, CL_DEVICE_TYPE_CPU, 1, &ctx->device_id, NULL);
        }
        else
        {
            printf("Error: Failed to create a device group!\n");
            return;
        }
    }

    // Create a compute context
    //
    ctx->context = clCreateContext(0, 1, &ctx->device_id, NULL, NULL, &err);
    if (!ctx->context)
    {
        printf("Error: Failed to create a compute context!\n");
        return;
    }

    // Create a command commands
    //
    ctx->commands = clCreateCommandQueue(ctx->context, ctx->device_id, 0, &err);
    if (!ctx->commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return;
    }

}

void load_program_raw(rcl_ctx* ctx, char* data,
                     char** kernels, unsigned int num_kernels,
                      rcl_program* program, char** macros, unsigned int num_macros)
{
    int err;

    char* fin_data = (char*) malloc(strlen(data));
    strcpy(fin_data, data);

    for(int i = 0; i < num_macros; i++)
    {
        int length = strlen(macros[i]);
        char* buf  = (char*) malloc(length+strlen(fin_data)+2);
        sprintf(buf, "%s\n%s", macros[i], fin_data);
        free(fin_data);
        fin_data = buf;
    }

    program->program = clCreateProgramWithSource(ctx->context, 1, (const char **) &fin_data, NULL, &err);
    if (!program->program)
    {
        printf("Error: Failed to create compute program!\n");
        return;
    }

    // Build the program executable
    //
    err = clBuildProgram(program->program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048*256];
        buffer[0] = '!';
        buffer[1] = '\0';


        printf("Error: Failed to build program executable!\n");
        printf("KERNEL:\n %s\nprogram done\n\n", fin_data);
        int n_err = clGetProgramBuildInfo(program->program, ctx->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        if(n_err != CL_SUCCESS)
        {
            printf("WTF the error had an error I hate this. err:%i\n",n_err);
        }
        printf("err code:%i\n %s\n", err, buffer);
        exit(1);
    }

    program->raw_kernels = malloc(sizeof(cl_kernel)*num_kernels);
    for(int i = 0; i < num_kernels; i++)
    {
        // Create the compute kernel in the program we wish to run
        //
        program->raw_kernels[i] = clCreateKernel(program->program, kernels[i], &err);
        if (!program->raw_kernels[i] || err != CL_SUCCESS)
        {
            printf("Error: Failed to create compute kernel!\n");
            exit(1);
        }
    }

    program->raw_data = data;

}

void load_program_url(rcl_ctx* ctx, char* url,
                     char** kernels, unsigned int num_kernels,
                      rcl_program* program, char** macros, unsigned int num_macros)
{
    char * buffer = 0;
    long length;
    FILE * f = fopen (url, "rb");

    if (f)
    {
        fseek (f, 0, SEEK_END);
        length = ftell (f);
        fseek (f, 0, SEEK_SET);
        buffer = malloc (length);
        if (buffer)
        {
            fread (buffer, 1, length, f);
        }
        fclose (f);
    }
    if (buffer)
    {
        buffer[length] = '\0';

        load_program_raw(ctx, buffer, kernels, num_kernels, program,
                         macros, num_macros);
    }

}

void test_sphere_raytracer(rcl_ctx* ctx, rcl_program* program,
        sphere* spheres, int num_spheres,
        uint32_t* bitmap, int width, int height)
{
    int err;

    static cl_mem tex;
    static cl_mem s_buf;
    static bool init = false;

    if(!init)
    {
        //New Texture
        tex = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY,
                                    width*height*4, NULL, &err);

        //Spheres
        s_buf = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float)*4*num_spheres, spheres, &err);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to create Sphere Buffer! %d\n", err);
            return;
        }
        init = true;
    }
    else
    {
        clEnqueueWriteBuffer (	ctx->commands,
                                s_buf,
                                CL_TRUE,
                                0,
                                sizeof(float)*4*num_spheres,
                                spheres,
                                0,
                                NULL,
                                NULL);
    }



    cl_kernel kernel = program->raw_kernels[0]; //just use the first one

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &tex);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &s_buf);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &width);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &height);


    size_t global;
    size_t local = 0;

    err = clGetKernelWorkGroupInfo(kernel, ctx->device_id, CL_KERNEL_WORK_GROUP_SIZE,
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
    global =  width*height;
    err = clEnqueueNDRangeKernel(ctx->commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel! %i\n",err);
        return;
    }


    clFinish(ctx->commands);
    //printf("STOPPING\n");

    err = clEnqueueReadBuffer(ctx->commands, tex, CL_TRUE, 0, width*height*4, bitmap, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
}
