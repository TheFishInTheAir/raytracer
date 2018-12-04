#pragma once
#include <stdint.h>
#include <parallel.h>
#include <CL/opencl.h>
#include <scene.h>

typedef struct _rt_ctx raytracer_context;

struct _rt_ctx{
    unsigned int width, height;

    float* ray_buffer;
    uint32_t* output_buffer;
    //uint32_t* fresh_frame_buffer;

    scene* stat_scene;

    //CL
    rcl_ctx* rcl;
    rcl_program* program;
    unsigned int ray_buffer_kernel_index;
    unsigned int ray_cast_kernel_index;
    unsigned int path_trace_kernel_index;
    unsigned int buffer_average_kernel_index;

    cl_mem cl_ray_buffer;
    cl_mem cl_output_buffer; //NOTE: not going to really be used in path tracing
    cl_mem cl_fresh_frame_buffer; //Only exists on GPU


    //TODO: add stuff
};

raytracer_context* raytracer_init(unsigned int width, unsigned int height,
                                  uint32_t* output_buffer, rcl_ctx* ctx);
void raytracer_prepass(raytracer_context*);
void raytracer_render(raytracer_context*);
void raytracer_refined_render(raytracer_context*);
void _raytracer_gen_ray_buffer(raytracer_context*);
void _raytracer_path_trace(raytracer_context*, int);
void _raytracer_average_buffers(raytracer_context*, int); //NOTE: DEPRECATED

void _raytracer_cast_rays(raytracer_context*); //NOTE: DEPRECATED
