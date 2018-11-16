#pragma once

typedef struct
{
    void (*start_func)();
    void (*loop_start_func)();
    void (*update_func)();
    void (*sleep_func)(int);
    void* (*get_bitmap_memory_func)();
    int  (*get_time_mili_func)();
    int  (*get_width_func)();
    int  (*get_height_func)();
    void (*start_thread_func)(void (*func)(void*), void* data);
} os_abs;

void os_start(os_abs);
void os_loop_start(os_abs);
void os_update(os_abs);
void os_sleep(os_abs, int);
void* os_get_bitmap_memory(os_abs);
int os_get_time_mili(os_abs);
int os_get_width(os_abs);
int os_get_height(os_abs);
void os_start_thread(os_abs, void (*func)(void*), void* data);
