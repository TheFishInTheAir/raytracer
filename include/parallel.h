#pragma once
#include <alignment_util.h>

#include <CL/opencl.h>
#include <geom.h>

#define MACRO_GEN(n, t, v,  i)                 \
    char n[64];                                \
    sprintf(n, "#define " #t, v);              \
    i++;                                       \


typedef struct _rt_ctx raytracer_context;

typedef struct
{
    cl_platform_id platform_id;
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue

    unsigned int simt_size;
    unsigned int num_simt_per_multiprocessor;
    unsigned int num_multiprocessors;
    unsigned int num_cores;

} rcl_ctx;

typedef struct
{
    cl_program program;
    cl_kernel* raw_kernels; //NOTE: not an ideal solution
    char*      raw_data;

} rcl_program;

typedef struct rcl_img_buf
{
    cl_mem buffer;
    cl_mem image;
    size_t size;
} rcl_img_buf;

void cl_info();
void create_context(rcl_ctx* context);

void load_program_raw(rcl_ctx* ctx, char* data, char** kernels, unsigned int num_kernels,
                      rcl_program* program, char** macros, unsigned int num_macros);
void load_program_url(rcl_ctx* ctx, char* url,  char** kernels, unsigned int num_kernels,
                      rcl_program* program, char** macros, unsigned int num_macros);

cl_mem gen_rgb_image(raytracer_context* rctx,
                     const unsigned int width,
                     const unsigned int height);
cl_mem gen_grayscale_buffer(raytracer_context* rctx,
                            const unsigned int width,
                            const unsigned int height);

cl_mem gen_1d_image(raytracer_context* rctx, size_t t, void* ptr);
rcl_img_buf gen_1d_image_buffer(raytracer_context* rctx, size_t t, void* ptr);

void retrieve_buf(raytracer_context* rctx, cl_mem g_buf, void* c_buf, size_t);
void zero_buffer(raytracer_context* rctx, cl_mem buf, size_t size);
void zero_buffer_img(raytracer_context* rctx, cl_mem buf, size_t element,
                 const unsigned int width,
                 const unsigned int height);

size_t get_workgroup_size(raytracer_context* rctx, cl_kernel kernel);
