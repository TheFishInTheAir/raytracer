#pragma once
#include <alignment_util.h>

#include <stdint.h>
#include <parallel.h>
#include <CL/opencl.h>
#include <scene.h>
#include <irradiance_cache.h>

#define SS_RAYTRACER 0
#define PATH_RAYTRACER 1
#define SPLIT_PATH_RAYTRACER 2

//Cheap, quick, and dirty way of managing kernels.
#define KERNELS {"cast_ray_test", "generate_rays", "path_trace",        \
                 "buffer_average", "f_buffer_average",                  \
                 "f_buffer_to_byte_buffer",                             \
                 "ic_screen_textures", "generate_discontinuity",        \
                 "float_average", "mip_single_upsample", "mip_upsample",\
                 "mip_upsample_scaled", "mip_single_upsample_scaled",   \
                 "mip_reduce", "blit_float_to_output",                  \
                 "blit_float3_to_output", "kdtree_intersection",        \
                 "kdtree_test_draw", "segmented_path_trace",            \
                 "f_buffer_to_byte_buffer_avg", "segmented_path_trace_init", \
                 "kdtree_ray_draw", "xorshift_batch"}
#define NUM_KERNELS 23
#define RAY_CAST_KRNL_INDX 0
#define RAY_BUFFER_KRNL_INDX 1
#define PATH_TRACE_KRNL_INDX 2
#define BUFFER_AVG_KRNL_INDX 3
#define F_BUFFER_AVG_KRNL_INDX 4
#define F_BUF_TO_BYTE_BUF_KRNL_INDX 5
#define IC_SCREEN_TEX_KRNL_INDX 6
#define IC_GEN_DISC_KRNL_INDX 7
#define IC_FLOAT_AVG_KRNL_INDX 8
#define IC_MIP_S_UPSAMPLE_KRNL_INDX 9
#define IC_MIP_UPSAMPLE_KRNL_INDX 10
#define IC_MIP_UPSAMPLE_SCALED_KRNL_INDX 11
#define IC_MIP_S_UPSAMPLE_SCALED_KRNL_INDX 12
#define IC_MIP_REDUCE_KRNL_INDX 13
#define BLIT_FLOAT_OUTPUT_INDX 14
#define BLIT_FLOAT3_OUTPUT_INDX 15
#define KDTREE_INTERSECTION_INDX 16
#define KDTREE_TEST_DRAW_INDX 17
#define SEGMENTED_PATH_TRACE_INDX 18
#define F_BUF_TO_BYTE_BUF_AVG_KRNL_INDX 19
#define SEGMENTED_PATH_TRACE_INIT_INDX 20
#define KDTREE_RAY_DRAW_INDX 21
#define XORSHIFT_BATCH_INDX 22

typedef struct _rt_ctx raytracer_context;

struct _rt_ctx
{
    unsigned int width, height;

    float* ray_buffer;
    uint32_t* output_buffer;

    scene* stat_scene;
    ic_context* ic_ctx;

    unsigned int block_size_y;
    unsigned int block_size_x;

    unsigned int event_stack[32];
    unsigned int event_position;

    bool render_complete;

    //CL
    rcl_ctx* rcl;
    rcl_program* program;

    cl_mem cl_ray_buffer;
    cl_mem cl_output_buffer;


    //TODO: seperate the basic integrator from the main context.
    vec4*  path_output_buffer;
    unsigned int num_samples;
    unsigned int current_sample;
    cl_mem cl_path_output_buffer;
    cl_mem cl_path_fresh_frame_buffer; //Only exists on GPU

};

raytracer_context* raytracer_init(unsigned int width, unsigned int height,
                                  uint32_t* output_buffer, rcl_ctx* ctx);

void raytracer_build(raytracer_context*);
void raytracer_prepass(raytracer_context*); //NOTE: Semantically more like a second part of the build
void raytracer_render(raytracer_context*);
void raytracer_refined_render(raytracer_context*);
void _raytracer_gen_ray_buffer(raytracer_context*);
void _raytracer_path_trace(raytracer_context*, unsigned int);
void _raytracer_average_buffers(raytracer_context*, unsigned int); //NOTE: DEPRECATED
void _raytracer_push_path(raytracer_context*);
void _raytracer_cast_rays(raytracer_context*); //NOTE: DEPRECATED
