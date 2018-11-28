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

    scene* stat_scene;

    //CL
    rcl_ctx* rcl;
    rcl_program* program;
    unsigned int ray_buffer_kernel_index;
    unsigned int ray_cast_kernel_index;

    cl_mem cl_ray_buffer;
    cl_mem cl_output_buffer;

    //TODO: add stuff
};

raytracer_context* raytracer_init(unsigned int width, unsigned int height,
                                  uint32_t* output_buffer, rcl_ctx* ctx);
void raytracer_prepass(raytracer_context*);
void raytracer_render(raytracer_context*);
void _raytracer_gen_ray_buffer(raytracer_context*);
void _raytracer_cast_rays(raytracer_context*); //TODO: do more path tracing stuff here
