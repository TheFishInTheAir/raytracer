#pragma once

struct _rt_ctx;

typedef struct ss_raytracer_context
{
    struct _rt_ctx* rctx; //General Raytracer Context
    bool up_to_date;
} ss_raytracer_context;


//TODO: create function table;

rt_vtable get_ss_raytracer_vtable();

ss_raytracer_context* init_ss_raytracer_context(struct _rt_ctx*);

void ss_raytracer_render(ss_raytracer_context*);
//void ss_raytracer_build(ss_raytracer_context*);
void ss_raytracer_prepass(ss_raytracer_context*);