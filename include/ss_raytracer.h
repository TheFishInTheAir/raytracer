#pragma once

struct _rt_ctx;

typedef struct ss_raytracer_context
{
    struct _rt_ctx* rctx; //General Raytracer Context
} ss_raytracer_context;


ss_raytracer_context* init_ss_raytracer_context(struct _rt_ctx*);

void ss_raytracer_render(ss_raytracer_context*);
