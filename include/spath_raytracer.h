#pragma once

struct _rt_ctx;


typedef struct spath_raytracer_context
{
    struct _rt_ctx* rctx; //General Raytracer Context
    bool up_to_date;

    unsigned int num_iterations;
    unsigned int current_iteration;
    bool render_complete;

    unsigned int start_time;

    unsigned int* random_buffer;

    cl_mem cl_path_output_buffer;
    cl_mem cl_path_ray_origin_buffer; //Only exists on GPU
    cl_mem cl_path_collision_result_buffer; //Only exists on GPU
    cl_mem cl_spath_progress_buffer; //Only exists on GPU
    cl_mem cl_path_origin_collision_result_buffer; //Only exists on GPU

    cl_mem cl_random_buffer; //Only exists on GPU


    cl_mem cl_bad_api_design_buffer;


} spath_raytracer_context;

spath_raytracer_context* init_spath_raytracer_context(struct _rt_ctx*);

void spath_raytracer_render(spath_raytracer_context*);

void spath_raytracer_prepass(spath_raytracer_context*);
