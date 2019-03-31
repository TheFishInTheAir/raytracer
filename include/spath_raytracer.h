#pragma once

struct _rt_ctx;


typedef struct spath_raytracer_context
{
    struct _rt_ctx* rctx; //General Raytracer Context
    bool up_to_date;

    unsigned int num_samples;
    unsigned int current_sample;
    bool render_complete;

    cl_mem cl_path_output_buffer;
    cl_mem cl_path_fresh_frame_buffer; //Only exists on GPU
    cl_mem cl_path_collision_result_buffer; //Only exists on GPU

    cl_mem cl_bad_api_design_buffer;


} spath_raytracer_context;

spath_raytracer_context* init_spath_raytracer_context(struct _rt_ctx*);

void spath_raytracer_render(spath_raytracer_context*);
//void ss_raytracer_build(ss_raytracer_context*);
void spath_raytracer_prepass(spath_raytracer_context*);
