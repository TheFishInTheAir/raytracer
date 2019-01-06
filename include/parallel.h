#pragma once
#include <alignment_util.h>

#include <CL/opencl.h>
#include <geom.h>
typedef struct _rt_ctx raytracer_context;

typedef struct
{
    cl_platform_id platform_id;
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue

} rcl_ctx;

typedef struct
{
    cl_program program;
    cl_kernel* raw_kernels; //NOTE: not a good solution
    char*      raw_data;

} rcl_program;

void cl_info();
void create_context(rcl_ctx* context);
void load_program_raw(rcl_ctx* ctx, char* data, char** kernels, unsigned int num_kernels,
                      rcl_program* program, char** macros, unsigned int num_macros);
void load_program_url(rcl_ctx* ctx, char* url,  char** kernels, unsigned int num_kernels,
                      rcl_program* program, char** macros, unsigned int num_macros);
void test_sphere_raytracer(rcl_ctx* ctx, rcl_program* program,
                           sphere* spheres, int num_spheres,
                           uint32_t* bitmap, int width, int height);
cl_mem gen_rgb_image(raytracer_context* rctx,
                     const unsigned int width,
                     const unsigned int height);
cl_mem gen_grayscale_buffer(raytracer_context* rctx,
                            const unsigned int width,
                            const unsigned int height);
cl_mem gen_1d_image(raytracer_context* rctx, size_t t, void* ptr);
void retrieve_buf(raytracer_context* rctx, cl_mem g_buf, void* c_buf, size_t);

void zero_buffer_img(raytracer_context* rctx, cl_mem buf, size_t element,
                 const unsigned int width,
                 const unsigned int height);
void zero_buffer(raytracer_context* rctx, cl_mem buf, size_t size);
size_t get_workgroup_size(raytracer_context* rctx, cl_kernel kernel);
