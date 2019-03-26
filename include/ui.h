#pragma once

struct _rt_ctx;


typedef struct ui_ctx
{
    struct _rt_ctx* rctx; //General Raytracer Context

} ui_ctx;

void web_server_start(void*);
