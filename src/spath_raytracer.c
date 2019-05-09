#include <spath_raytracer.h>
#include <kdtree.h>
#include <raytracer.h>
#include <stdlib.h>
//#include <windows.h>
typedef struct W_ALIGN(16) spath_progress
{
    unsigned int sample_num;
    unsigned int bounce_num;
    vec3 mask;
    vec3 accum_color;
} U_ALIGN(16) spath_progress; //NOTE: space for two more 32 bit dudes


void bad_buf_update(spath_raytracer_context* sprctx)
{
    int err;

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

    err = clEnqueueWriteBuffer(sprctx->rctx->rcl->commands, sprctx->cl_bad_api_design_buffer, CL_TRUE,
                               0, (4*4+1)*sizeof(float),bad_buf,
                               0, NULL, NULL);
    ASRT_CL("Error Creating OpenCL BAD API DESIGN! Buffer.");

    err = clFinish(sprctx->rctx->rcl->commands);
    ASRT_CL("Something happened while waiting for copy to finish");
}

spath_raytracer_context* init_spath_raytracer_context(struct _rt_ctx* rctx)
{
    spath_raytracer_context* sprctx = (spath_raytracer_context*) malloc(sizeof(spath_raytracer_context));
    sprctx->rctx = rctx;
    sprctx->up_to_date = false;

    int err;
    printf("Generating Split Pathtracer Buffers...\n");


    sprctx->cl_path_output_buffer = clCreateBuffer(rctx->rcl->context,
                                                   CL_MEM_READ_WRITE,
                                                   rctx->width*rctx->height*sizeof(vec4),
                                                   NULL, &err);
    ASRT_CL("Error Creating OpenCL Split Path Tracer Output Buffer.");

    sprctx->cl_path_ray_origin_buffer = clCreateBuffer(rctx->rcl->context,
                                                             CL_MEM_READ_WRITE,
                                                             rctx->width*rctx->height*
                                                             sizeof(ray),
                                                             NULL, &err);
    ASRT_CL("Error Creating OpenCL Split Path Tracer Collision Result Buffer.");

    sprctx->cl_path_collision_result_buffer = clCreateBuffer(rctx->rcl->context,
                                                             CL_MEM_READ_WRITE,
                                                             rctx->width*rctx->height*
                                                             sizeof(kd_tree_collision_result),
                                                             NULL, &err);
    ASRT_CL("Error Creating OpenCL Split Path Tracer Collision Result Buffer.");

    sprctx->cl_path_origin_collision_result_buffer = clCreateBuffer(rctx->rcl->context,
                                                                    CL_MEM_READ_WRITE,
                                                                    rctx->width*rctx->height*
                                                                    sizeof(kd_tree_collision_result),
                                                                    NULL, &err);
    ASRT_CL("Error Creating OpenCL Split Path Tracer ORIGIN Collision Result Buffer.");

    sprctx->cl_random_buffer = clCreateBuffer(rctx->rcl->context,
                                              CL_MEM_READ_WRITE,
                                              rctx->width * rctx->height * sizeof(unsigned int),
                                              NULL, &err);
    ASRT_CL("Error Creating OpenCL Random Buffer.");

    sprctx->random_buffer = (unsigned int*) malloc(rctx->width * rctx->height * sizeof(unsigned int));


    sprctx->cl_spath_progress_buffer = clCreateBuffer(rctx->rcl->context,
                                                      CL_MEM_READ_WRITE,
                                                      rctx->width*rctx->height*
                                                      sizeof(spath_progress),
                                                      NULL, &err);
    zero_buffer(rctx, sprctx->cl_spath_progress_buffer, rctx->width*rctx->height*sizeof(spath_progress));

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

void spath_raytracer_update_random(spath_raytracer_context* sprctx)
{
    for(int i= 0; i < sprctx->rctx->width*sprctx->rctx->height; i++)
        sprctx->random_buffer[i] = rand();

    int err;

    err =  clEnqueueWriteBuffer (	sprctx->rctx->rcl->commands,
                                    sprctx->cl_random_buffer,
                                    CL_TRUE, 0,
                                    sprctx->rctx->width * sprctx->rctx->height * sizeof(unsigned int),
                                    sprctx->random_buffer,
                                    0, NULL, NULL);
    ASRT_CL("Couldn't Push Random Buffer to GPU.");
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

    clSetKernelArg(kernel, 6, sizeof(cl_mem), &sprctx->rctx->stat_scene->kdt->cl_kd_tree_buffer);
    //NOTE: WILL NOT WORK WITH ALL SITUATIONS:
    unsigned int num_rays = sprctx->rctx->width*sprctx->rctx->height;
    clSetKernelArg(kernel, 7, sizeof(unsigned int), &num_rays);




    size_t global[1] = {sprctx->rctx->rcl->num_cores * 16};//sprctx->rctx->rcl->simt_size; sprctx->rctx->rcl->num_simt_per_multiprocessor};//ok I give up with the peristent threading.
    size_t local[1]  = {sprctx->rctx->rcl->simt_size};//sprctx->rctx->rcl->simt_size; sprctx->rctx->rcl->num_simt_per_multiprocessor};// * sprctx->rctx->rcl->num_simt_per_multiprocessor};//sprctx->rctx->rcl->simt_size; sprctx->rctx->rcl->num_simt_per_multiprocessor
    err = clEnqueueNDRangeKernel(sprctx->rctx->rcl->commands, kernel, 1,
                                 NULL, global, local, 0, NULL, NULL);
    ASRT_CL("Failed to execute kd tree traversal kernel");

    err = clFinish(sprctx->rctx->rcl->commands);
    ASRT_CL("Something happened while executing kd tree traversal kernel");

}

void spath_raytracer_ray_test(spath_raytracer_context* sprctx)
{
    int err;

    cl_kernel kernel = sprctx->rctx->program->raw_kernels[KDTREE_RAY_DRAW_INDX];

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sprctx->rctx->cl_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &sprctx->rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &sprctx->rctx->width);
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

    clSetKernelArg(kernel, 7, sizeof(unsigned int), &sprctx->rctx->width);
    //NOTE: WILL NOT WORK WITH ALL SITUATIONS:
    unsigned int num_rays = sprctx->rctx->width*sprctx->rctx->height;


    size_t global[1] = {num_rays};

    err = clEnqueueNDRangeKernel(sprctx->rctx->rcl->commands, kernel, 1,
                                 NULL, global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kd tree traversal kernel");

    err = clFinish(sprctx->rctx->rcl->commands);
    ASRT_CL("Something happened while executing kd tree test kernel");

    err = clEnqueueReadBuffer(sprctx->rctx->rcl->commands, sprctx->rctx->cl_output_buffer, CL_TRUE, 0,
                              sprctx->rctx->width*sprctx->rctx->height*sizeof(int),
                              sprctx->rctx->output_buffer, 0, NULL, NULL );
    ASRT_CL("Failed to read output array");
}

void spath_raytracer_xor_rng(spath_raytracer_context* sprctx)
{
    int err;
    cl_kernel kernel = sprctx->rctx->program->raw_kernels[XORSHIFT_BATCH_INDX];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sprctx->cl_random_buffer);

    size_t global = sprctx->rctx->width*sprctx->rctx->height;

    err = clEnqueueNDRangeKernel(sprctx->rctx->rcl->commands, kernel, 1, NULL,
                                 &global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kernel");
    err = clFinish(sprctx->rctx->rcl->commands);
    ASRT_CL("Something happened while waiting for kernel to finish");
}

void spath_raytracer_avg_to_out(spath_raytracer_context* sprctx)
{
    int err;
    int useless = 0;
    cl_kernel kernel = sprctx->rctx->program->raw_kernels[F_BUF_TO_BYTE_BUF_AVG_KRNL_INDX];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sprctx->rctx->cl_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &sprctx->cl_path_output_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sprctx->cl_spath_progress_buffer);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &sprctx->rctx->width);

    clSetKernelArg(kernel, 4, sizeof(unsigned int), &useless);


    size_t global = sprctx->rctx->width*sprctx->rctx->height;

    err = clEnqueueNDRangeKernel(sprctx->rctx->rcl->commands, kernel, 1, NULL,
                                 &global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kernel");
    err = clFinish(sprctx->rctx->rcl->commands);
    ASRT_CL("Something happened while waiting for kernel to finish");

    err = clEnqueueReadBuffer(sprctx->rctx->rcl->commands, sprctx->rctx->cl_output_buffer, CL_TRUE, 0,
                              sprctx->rctx->width*sprctx->rctx->height*sizeof(int),
                              sprctx->rctx->output_buffer, 0, NULL, NULL );
    ASRT_CL("Failed to read output array");
}


void spath_raytracer_trace_init(spath_raytracer_context* sprctx)
{
    int err;
    unsigned int random_value_WACKO = rand();
    cl_kernel kernel = sprctx->rctx->program->raw_kernels[SEGMENTED_PATH_TRACE_INIT_INDX];

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sprctx->cl_path_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &sprctx->rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sprctx->cl_path_ray_origin_buffer);

    clSetKernelArg(kernel, 3, sizeof(cl_mem),  &sprctx->cl_path_collision_result_buffer);
    clSetKernelArg(kernel, 4,  sizeof(cl_mem), &sprctx->cl_path_origin_collision_result_buffer);

//SPATH DATA
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &sprctx->cl_spath_progress_buffer);


    clSetKernelArg(kernel, 6, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_material_buffer);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_buffer);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_index_buffer.image);
    clSetKernelArg(kernel, 9, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_vert_buffer.image);
    clSetKernelArg(kernel, 10, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_nrml_buffer.image);

    clSetKernelArg(kernel, 11, sizeof(unsigned int), &sprctx->rctx->width);
    clSetKernelArg(kernel, 12, sizeof(unsigned int), &random_value_WACKO);


    size_t global[1] = {sprctx->rctx->width*sprctx->rctx->height};

    err = clEnqueueNDRangeKernel(sprctx->rctx->rcl->commands, kernel, 1,
                                 NULL, global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kd tree traversal kernel");

    err = clFinish(sprctx->rctx->rcl->commands);
    ASRT_CL("Something happened while executing kd init kernel");

}

void spath_raytracer_trace(spath_raytracer_context* sprctx)
{
    int err;
    unsigned int random_value_WACKO = rand();// sprctx->current_iteration; //TODO: make an actual random number
    cl_kernel kernel = sprctx->rctx->program->raw_kernels[SEGMENTED_PATH_TRACE_INDX];

    unsigned int karg = 0;
    clSetKernelArg(kernel, karg++, sizeof(cl_mem), &sprctx->cl_path_output_buffer);
    clSetKernelArg(kernel, karg++, sizeof(cl_mem), &sprctx->rctx->cl_ray_buffer);
    clSetKernelArg(kernel, karg++, sizeof(cl_mem), &sprctx->cl_path_ray_origin_buffer);

    clSetKernelArg(kernel, karg++, sizeof(cl_mem),  &sprctx->cl_path_collision_result_buffer);
    clSetKernelArg(kernel, karg++,  sizeof(cl_mem), &sprctx->cl_path_origin_collision_result_buffer);
    clSetKernelArg(kernel, karg++, sizeof(cl_mem), &sprctx->cl_spath_progress_buffer);

    clSetKernelArg(kernel, karg++, sizeof(cl_mem), &sprctx->cl_random_buffer);

    clSetKernelArg(kernel, karg++, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_material_buffer);
    clSetKernelArg(kernel, karg++, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_buffer);
    clSetKernelArg(kernel, karg++, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_index_buffer.image);
    clSetKernelArg(kernel, karg++, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_vert_buffer.image);
    clSetKernelArg(kernel, karg++, sizeof(cl_mem), &sprctx->rctx->stat_scene->cl_mesh_nrml_buffer.image);

    clSetKernelArg(kernel, karg++, sizeof(unsigned int), &sprctx->rctx->width);
    clSetKernelArg(kernel, karg++, sizeof(unsigned int), &random_value_WACKO);

    size_t global[1] = {sprctx->rctx->width*sprctx->rctx->height};

    err = clEnqueueNDRangeKernel(sprctx->rctx->rcl->commands, kernel, 1,
                                 NULL, global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to execute kd tree traversal kernel");

    err = clFinish(sprctx->rctx->rcl->commands);
    ASRT_CL("Something happened while executing kd tree traversal kernel");

}

void spath_raytracer_render(spath_raytracer_context* sprctx)
{
    static int tbottle = 0;
    //Sleep(5000);
    if((sprctx->current_iteration+1)%50 == 0)
        int t1 = os_get_time_mili(abst);

    //spath_raytracer_update_random(sprctx);
    spath_raytracer_xor_rng(sprctx);
    sprctx->current_iteration++;
    if(sprctx->current_iteration>sprctx->num_iterations)
    {
        if(!sprctx->render_complete)
            printf("Render took: %d ms", (unsigned int) os_get_time_mili(abst)-sprctx->start_time);
        sprctx->render_complete = true;

        return;
    }

    //spath_raytracer_ray_test(sprctx);


    bad_buf_update(sprctx);

    if(sprctx->current_iteration%50 == 0)
        int t2 = os_get_time_mili(abst);

    spath_raytracer_kd_collision(sprctx);
    if(sprctx->current_iteration%50 == 0)
        int t3 = os_get_time_mili(abst);

    spath_raytracer_trace(sprctx);
    if(sprctx->current_iteration%50 == 0)
        int t4 = os_get_time_mili(abst);

    if(sprctx->current_iteration%50 == 0)
        spath_raytracer_avg_to_out(sprctx);

    if(sprctx->current_iteration%50 == 0)
        int t5 = os_get_time_mili(abst);

    if(sprctx->current_iteration%50 == 0)
        printf("num_gen: %d, collision: %d, trace: %d, draw: %d, time_since: %d, total: %d    %d.%d/%d    %d:%d:%d\n",
               t2-t1, t3-t2, t4-t3, t5-t4, t1-tbottle, t5-tbottle,
               sprctx->current_iteration/4, sprctx->current_iteration%4, sprctx->num_iterations/4,
               ((t5-sprctx->start_time)/1000)/60, ((t5-sprctx->start_time)/1000)%60, (t5-sprctx->start_time)%1000);
    //spath_raytracer_kd_test(sprctx);
    tbottle = os_get_time_mili(abst);
}

void spath_raytracer_prepass(spath_raytracer_context* sprctx)
{
    printf("Starting Split Path Raytracer Prepass. \n");
    sprctx->render_complete = false;
    sprctx->num_iterations = 2048*4;//arbitrary default
    srand((unsigned int)os_get_time_mili(abst));
    sprctx->start_time = (unsigned int) os_get_time_mili(abst);
    bad_buf_update(sprctx);


    zero_buffer(sprctx->rctx, sprctx->cl_path_output_buffer,
                sprctx->rctx->width*sprctx->rctx->height*sizeof(vec4));

    raytracer_prepass(sprctx->rctx);

    sprctx->current_iteration = 0;
    zero_buffer(sprctx->rctx, sprctx->cl_spath_progress_buffer,
                sprctx->rctx->width*sprctx->rctx->height*sizeof(spath_progress));
    _raytracer_gen_ray_buffer(sprctx->rctx);



    spath_raytracer_kd_collision(sprctx);

    spath_raytracer_trace_init(sprctx);

    spath_raytracer_update_random(sprctx);

    zero_buffer(sprctx->rctx, sprctx->rctx->cl_ray_buffer,
                sprctx->rctx->width*sprctx->rctx->height*sizeof(ray));

    printf("Finished Split Path Raytracer Prepass. \n");
}
