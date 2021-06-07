#include <CL/opencl.h>
#include <raytracer.h>
//Parallel util.

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
    cl_uint recommendedWorkgroupSize = 0;

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
            printf("%i.%d. Device: %s\n", i, j+1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %i.%d.%d Hardware version: %s\n", i, j+1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %i.%d.%d Software version: %s\n", i, j+1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %i.%d.%d OpenCL C version: %s\n", i, j+1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %i.%d.%d Parallel compute units: %d\n", i,  j+1, 4, maxComputeUnits);

            size_t max_work_group_size;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            sizeof(max_work_group_size), &max_work_group_size, NULL); //NOTE: just reuse var
            printf(" %i.%d.%d Max work group size: %zu\n", i,  j+1, 4, max_work_group_size);

            clGetDeviceInfo(devices[j], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            sizeof(recommendedWorkgroupSize), &recommendedWorkgroupSize, NULL);
            printf(" %i.%d.%d Recommended work group size: %d\n", i,  j+1, 4, recommendedWorkgroupSize);

        }

        free(devices);

    }
    printf("\n");
    free(platforms);
    return;
}
void pfn_notify (
    const char *errinfo,
    const void *private_info,
    size_t cb,
    void *user_data)
{
    fprintf(stderr, "\n--\nOpenCL ERROR: %s\n--\n", errinfo);
    fflush(stderr);
}
void create_context(rcl_ctx* ctx)
{
    int err = CL_SUCCESS;


    unsigned int num_of_platforms;

    if (clGetPlatformIDs(0, NULL, &num_of_platforms) != CL_SUCCESS)
    {
        printf("Error: Unable to get platform_id\n");
        exit(1);
    }
    cl_platform_id *platform_ids = malloc(num_of_platforms*sizeof(cl_platform_id));
    if (clGetPlatformIDs(num_of_platforms, platform_ids, NULL) != CL_SUCCESS)
    {
        printf("Error: Unable to get platform_id\n");
        exit(1);
    }
    bool found = false;
    for(int i=0; i<num_of_platforms; i++)
    {
        cl_device_id device_ids[16];
        unsigned int num_devices = 0;

        //arbitrarily choosing 16 as the max gpus on a platform.
        if(clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 16, device_ids, &num_devices) == CL_SUCCESS)
        {

            for(int j = 0; j < num_devices; j++)
            {
                char* value;
                size_t valueSize;
                clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
                value = (char*) malloc(valueSize);
                clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, valueSize, value, NULL);
                if(value[0]=='H'&&value[1]=='D') //janky but whatever
                {
                    printf("WARNING: Skipping over '%s' during device selection\n", value);
                    free(value);
                    continue;
                }
                free(value);

                found = true;
                ctx->platform_id = platform_ids[i];
                ctx->device_id = device_ids[j];
                break;
            }
        }
        if(found)
            break;
    }
    if(!found){
        printf("Error: Unable to get a GPU device_id\n");
        exit(1);
    }
    char* value;
    size_t valueSize;
    clGetDeviceInfo(ctx->device_id, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(ctx->device_id, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Selected device: %s\n", value);
    free(value);

    // Create a compute context
    //
    ctx->context = clCreateContext(0, 1, &ctx->device_id, &pfn_notify, NULL, &err);
    if (!ctx->context)
    {
        printf("Error: Failed to create a compute context!\n");
        exit(1);
    }

    // Create a command commands
    //
    ctx->commands = clCreateCommandQueue(ctx->context, ctx->device_id, 0, &err);
    if (!ctx->commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return;
    }
    ASRT_CL("Failed to Initialise OpenCL");

    { // num compute cores
        unsigned int id = 0;
        clGetDeviceInfo(ctx->device_id, CL_DEVICE_VENDOR_ID, sizeof(unsigned int), &id, NULL);
        switch(id)
        {
        case(0x10DE): //NVIDIA
        {
            unsigned int warp_size;
            unsigned int compute_capability;
            unsigned int num_sm;
            unsigned int warps_per_sm;
            clGetDeviceInfo(ctx->device_id, CL_DEVICE_WARP_SIZE_NV, //warp size
                            sizeof(unsigned int), &warp_size, NULL);
            clGetDeviceInfo(ctx->device_id, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, //compute capability
                            sizeof(unsigned int), &compute_capability, NULL);
            clGetDeviceInfo(ctx->device_id, CL_DEVICE_MAX_COMPUTE_UNITS, //number of stream multiprocessors
                            sizeof(unsigned int), &num_sm, NULL);

            switch(compute_capability)
            { //nvidia skipped 4 btw lol
            case 2: warps_per_sm = 1; break; //FERMI  (GK104/GK110)
            case 3: warps_per_sm = 6; break; //KEPLER (GK104/GK110) NOTE: ONLY 4 WARP SCHEDULERS THOUGH!
            case 5: warps_per_sm = 4; break; //Maxwell
            case 6: warps_per_sm = 4; break; //Pascal is confusing because the sms vary a lot. GP100 is 2, but GP104 and GP106 have 4
            case 7: warps_per_sm = 2; break; //Volta/Turing Might not be correct(NOTE: 16 FP32 PER CORE?)
            }

            printf("NVIDIA INFO: SM: %d,  WARP SIZE: %d, COMPUTE CAPABILITY: %d, WARPS PER SM: %d, TOTAL STREAM PROCESSORS: %d\n\n",
                   num_sm, warp_size, compute_capability, warps_per_sm, warps_per_sm*warp_size*num_sm);
            ctx->simt_size = warp_size;
            ctx->num_simt_per_multiprocessor = warps_per_sm;
            ctx->num_multiprocessors = num_sm;
            ctx->num_cores = warps_per_sm*warp_size*num_sm;
            break;
        }
        case(0x1002): //AMD
        {
            printf("AMD GPU INFO NOT SUPPORTED YET!\n");
            break;
        }
        case(0x8086): //INTEL
        {
            printf("INTEL INFO NOT SUPPORTED YET!\n");
            break;
        }
        default: //APPLE is really bad and doesn't return the correct vendor id. This is a temporary fix
        {        //Just going to manually enter in data.
                printf("WARNING: Unknown Device Manufacturer %u (%04X)\n", id, id);
                unsigned int warp_size;
                unsigned int compute_capability;
                unsigned int num_sm;
                unsigned int warps_per_sm = 6; //my laptop uses kepler
                clGetDeviceInfo(ctx->device_id, CL_DEVICE_WARP_SIZE_NV, //warp size NOT WORKING ON OSX
                                sizeof(unsigned int), &warp_size, NULL);
                warp_size = 32;
                clGetDeviceInfo(ctx->device_id, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, //compute capability
                                sizeof(unsigned int), &compute_capability, NULL);
                clGetDeviceInfo(ctx->device_id, CL_DEVICE_MAX_COMPUTE_UNITS, //number of stream multiprocessors
                                sizeof(unsigned int), &num_sm, NULL);
                
                printf("ASSUMING NVIDIA.\nNVIDIA INFO: SM: %d,  WARP SIZE: %d, COMPUTE CAPABILITY: %d, WARPS PER SM: %d, TOTAL STREAM PROCESSORS: %d\n\n",
                       num_sm, warp_size, compute_capability, warps_per_sm, warps_per_sm*warp_size*num_sm);
                ctx->simt_size = warp_size;
                ctx->num_simt_per_multiprocessor = warps_per_sm;
                ctx->num_multiprocessors = num_sm;
                ctx->num_cores = warps_per_sm*warp_size*num_sm;
                
                break;
            }
        }

    }

}

cl_mem gen_rgb_image(raytracer_context* rctx,
                     const unsigned int width,
                     const unsigned int height)
{
    cl_image_desc cl_standard_descriptor;
    cl_image_format     cl_standard_format;
    cl_standard_format.image_channel_order     = CL_RGBA;
    cl_standard_format.image_channel_data_type = CL_FLOAT;

    cl_standard_descriptor.image_type = CL_MEM_OBJECT_IMAGE2D;
    cl_standard_descriptor.image_width = width==0  ? rctx->width  : width;
    cl_standard_descriptor.image_height = height==0 ? rctx->height : height;
    cl_standard_descriptor.image_depth  = 0;
    cl_standard_descriptor.image_array_size  = 0;
    cl_standard_descriptor.image_row_pitch  = 0;
    cl_standard_descriptor.num_mip_levels = 0;
    cl_standard_descriptor.num_samples = 0;
    cl_standard_descriptor.buffer = NULL;

    int err;

    cl_mem img = clCreateImage(rctx->rcl->context,
                                CL_MEM_READ_WRITE,
                                &cl_standard_format,
                               &cl_standard_descriptor,
                                NULL,
                                &err);
    ASRT_CL("Couldn't Create OpenCL Texture");
    return img;
}

rcl_img_buf gen_1d_image_buffer(raytracer_context* rctx, size_t t, void* ptr)
{
    int err = CL_SUCCESS;


    rcl_img_buf ib;
    ib.size = t;

    ib.buffer = clCreateBuffer(rctx->rcl->context,
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               t,
                               ptr,
                               &err);
    ASRT_CL("Error Creating OpenCL ImageBuffer Buffer");


    cl_image_desc cl_standard_descriptor;
    cl_image_format     cl_standard_format;
    cl_standard_format.image_channel_order     = CL_RGBA;
	cl_standard_format.image_channel_data_type = CL_FLOAT; //prob should be float

    cl_standard_descriptor.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
	cl_standard_descriptor.image_width = t/4 == 0 ? 1 : t/sizeof(float)/4;
    cl_standard_descriptor.image_height = 0;
    cl_standard_descriptor.image_depth  = 0;
    cl_standard_descriptor.image_array_size  = 0;
    cl_standard_descriptor.image_row_pitch  = 0;
	cl_standard_descriptor.image_slice_pitch = 0;
    cl_standard_descriptor.num_mip_levels = 0;
    cl_standard_descriptor.num_samples = 0;
    cl_standard_descriptor.buffer = ib.buffer;


    ib.image = clCreateImage(rctx->rcl->context,
                             0,
                             &cl_standard_format,
                             &cl_standard_descriptor,
                             NULL,//ptr,
                             &err);
    ASRT_CL("Error Creating OpenCL ImageBuffer Image");

    return ib;
}
cl_mem gen_1d_image(raytracer_context* rctx, size_t t, void* ptr)
{

    cl_image_desc cl_standard_descriptor;
    cl_image_format     cl_standard_format;
    cl_standard_format.image_channel_order     = CL_RGBA;
	cl_standard_format.image_channel_data_type = CL_FLOAT; //prob should be float

    cl_standard_descriptor.image_type = CL_MEM_OBJECT_IMAGE1D;
	cl_standard_descriptor.image_width = t/4 == 0 ? 1 : t/sizeof(float)/4;// t / 4 == 0 ? 1 : t / 4; //what?
    cl_standard_descriptor.image_height = 0;
    cl_standard_descriptor.image_depth  = 0;
    cl_standard_descriptor.image_array_size  = 0;
    cl_standard_descriptor.image_row_pitch  = 0;
	cl_standard_descriptor.image_slice_pitch = 0;
    cl_standard_descriptor.num_mip_levels = 0;
    cl_standard_descriptor.num_samples = 0;
    cl_standard_descriptor.buffer = NULL;


    int err = CL_SUCCESS;


    cl_mem img = clCreateImage(rctx->rcl->context,
                               CL_MEM_READ_WRITE | (/*ptr == NULL ? 0 :*/ CL_MEM_COPY_HOST_PTR),
                               &cl_standard_format,
                               &cl_standard_descriptor,
                               ptr,
                               &err);
    ASRT_CL("Couldn't Create OpenCL Texture");
    return img;
}
cl_mem gen_grayscale_buffer(raytracer_context* rctx,
                            const unsigned int width,
                            const unsigned int height)
{
    int err;

    cl_mem buf = clCreateBuffer(rctx->rcl->context, CL_MEM_READ_WRITE,
                                 (width==0  ? rctx->width  : width)*
                                 (height==0 ? rctx->height : height)*
                                 sizeof(float),
                                 NULL, &err);
    ASRT_CL("Couldn't Create OpenCL Float Buffer Image");
    return buf;
}

void retrieve_image(raytracer_context* rctx, cl_mem g_buf, void* c_buf,
                    const unsigned int width,
                    const unsigned int height)
{
    int err;
    size_t origin[3] = {0,0,0};
    size_t region[3] = {(width==0 ? rctx->width : width),
                        (height==0 ? rctx->height : height),
                        1};
    err = clEnqueueReadImage (rctx->rcl->commands,
                              g_buf,
                              CL_TRUE,
                              origin,
                              region,
                              0,
                              0,
                              c_buf,
                              0,
                              0,
                              NULL);
    ASRT_CL("Failed to retrieve Opencl Image");
}

void retrieve_buf(raytracer_context* rctx, cl_mem g_buf, void* c_buf, size_t size)
{
    int err;
    err = clEnqueueReadBuffer(rctx->rcl->commands, g_buf, CL_TRUE, 0,
                              size, c_buf,
                              0, NULL, NULL );
    ASRT_CL("Failed to retrieve Opencl Buffer");
}

void zero_buffer(raytracer_context* rctx, cl_mem buf, size_t size)
{
    int err;
    char pattern = 0;
    err =  clEnqueueFillBuffer (rctx->rcl->commands,
                                buf,
                                &pattern, 1 ,0,
                                size,
                                0, NULL, NULL);
    ASRT_CL("Couldn't Zero OpenCL Buffer");
}
void zero_buffer_img(raytracer_context* rctx, cl_mem buf, size_t element,
                 const unsigned int width,
                 const unsigned int height)
{
    int err;

    char pattern = 0;
    err =  clEnqueueFillBuffer (rctx->rcl->commands,
                                buf,
                                &pattern, 1 ,0,
                                (width==0  ? rctx->width  : width)*
                                (height==0 ? rctx->height : height)*
                                element,
                                0, NULL, NULL);
    ASRT_CL("Couldn't Zero OpenCL Buffer");
}
size_t get_workgroup_size(raytracer_context* rctx, cl_kernel kernel)
{
    int err;
    size_t local = 0;
    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id,
                                   CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(local), &local, NULL);
    ASRT_CL("Failed to Retrieve Kernel Work Group Info");
    return local;
}


//This is a quick implementation, not concerned about load times right now.
void load_program_raw(rcl_ctx* ctx, char* data,
                     char** kernels, unsigned int num_kernels,
                      rcl_program* program, char** macros, unsigned int num_macros)
{
    int err;

    char* macro_buf = malloc(1); //garbage just so free doesn't break
    macro_buf[0] = '\0';
    for(int i = 0; i < num_macros; i++)
    {
        int length = strlen(macros[i]);
        char* buf  = (char*) malloc(length+strlen(macro_buf)+2);
        sprintf(buf, "%s\n%s", macro_buf, macros[i]);

        free(macro_buf);
        macro_buf = buf;
    }
    
    char* fin_data = (char*) malloc(strlen(data)+strlen(macro_buf)+1);

    fin_data[0] = '\0';
    strcpy(fin_data, macro_buf);
    strcpy(fin_data+strlen(macro_buf), data);


    program->program = clCreateProgramWithSource(ctx->context, 1, (const char **) &fin_data, NULL, &err);
    if (!program->program)
    {
        printf("Error: Failed to create compute program!\n");
        exit(1);
    }

    // Build the program executable
    //
    err = clBuildProgram(program->program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048*25];
        buffer[0] = '!';
        buffer[1] = '\0';


        printf("Error: Failed to build program executable!\n");
        printf("KERNEL:\n %s\nprogram done\n", fin_data);
        int n_err = clGetProgramBuildInfo(program->program, ctx->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        if(n_err != CL_SUCCESS)
        {
            printf("The error had an error, I hate this. err:%i\n",n_err);
        }
        printf("err code:%i\n %s\n", err, buffer);
        exit(1);
    }
	else
	{
		size_t len;
		char buffer[2048 * 25];
		buffer[0] = '!';
		buffer[1] = '\0';
		int n_err = clGetProgramBuildInfo(program->program, ctx->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		if (n_err != CL_SUCCESS)
		{
			printf("The error had an error, I hate this. err:%i\n", n_err);
		}
		printf("Build info: %s\n", buffer);
	}

    program->raw_kernels = malloc(sizeof(cl_kernel)*num_kernels);
    for(int i = 0; i < num_kernels; i++)
    {
        // Create the compute kernel in the program we wish to run
        //

        program->raw_kernels[i] = clCreateKernel(program->program, kernels[i], &err);
        if (!program->raw_kernels[i] || err != CL_SUCCESS)
        {
            printf("Error: Failed to create compute kernel! %s\n", kernels[i]);
            exit(1);
        }

    }

    program->raw_data = fin_data;

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
        buffer = malloc (length+2);
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
