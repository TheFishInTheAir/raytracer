#pragma once

struct _rt_ctx;

typedef struct path_raytracer_context
{
    struct _rt_ctx* rctx; //General Raytracer Context
    bool up_to_date;

    unsigned int num_samples;
    unsigned int current_sample;
    bool render_complete;

    cl_mem cl_path_output_buffer; //TODO: put in path tracer file
    cl_mem cl_path_fresh_frame_buffer; //Only exists on GPU TODO: put in path tracer file.


} path_raytracer_context;


//TODO: create function table;

//rt_vtable get_path_raytracer_vtable();

path_raytracer_context* init_path_raytracer_context(struct _rt_ctx*);

void path_raytracer_render(path_raytracer_context*);
//void ss_raytracer_build(ss_raytracer_context*);
void path_raytracer_prepass(path_raytracer_context*);
