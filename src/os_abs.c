#include <os_abs.h>

void os_start(os_abs abs)
{
    (*abs.start_func)();
}

void os_loop_start(os_abs abs)
{
    (*abs.loop_start_func)();
}

void os_update(os_abs abs)
{
    (*abs.update_func)();
}

void os_sleep(os_abs abs, int num)
{
    (*abs.sleep_func)(num);
}

void* os_get_bitmap_memory(os_abs abs)
{
    return (*abs.get_bitmap_memory_func)();
}

void os_draw_weird(os_abs abs)
{
    (*abs.draw_weird)();
}

int os_get_time_mili(os_abs abs)
{
    return (*abs.get_time_mili_func)();
}

int os_get_width(os_abs abs)
{
    return (*abs.get_width_func)();
}

int os_get_height(os_abs abs)
{
    return (*abs.get_height_func)();
}

void os_start_thread(os_abs abs, void (*func)(void*), void* data)
{
    (*abs.start_thread_func)(func, data);
}
