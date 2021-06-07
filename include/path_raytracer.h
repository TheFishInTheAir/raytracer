#pragma once

struct _rt_ctx;

typedef struct path_raytracer_context
{
    struct _rt_ctx* rctx; //General Raytracer Context
    bool up_to_date;

    unsigned int num_samples;
    unsigned int current_sample;
    bool render_complete;
    int start_time;

    cl_mem cl_path_output_buffer;
    cl_mem cl_path_fresh_frame_buffer; //Only exists on GPU


} path_raytracer_context;

path_raytracer_context* init_path_raytracer_context(struct _rt_ctx*);

void path_raytracer_render(path_raytracer_context*);
void path_raytracer_prepass(path_raytracer_context*);
