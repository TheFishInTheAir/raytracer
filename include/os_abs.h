#pragma once

struct os_abs
{
    void (*start_func)();
    void (*loop_start_func)();
    void (*update_func)();
    void (*sleep_func)(int);
    void (*get_bitmap_memory_func)();
    int  (*get_time_mili_func)();
    int  (*get_width_func)();
    int  (*get_height_func)();
    void (*start_thread_func)(void (*func)(void*), void* data);

}
