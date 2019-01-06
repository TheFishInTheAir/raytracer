#pragma once
#include <time.h>
#include <os_abs.h>

os_abs init_osx_abs();

void osx_start();
void osx_loop_start();
void osx_enqueue_update();
void osx_sleep(int miliseconds);
void* osx_get_bitmap_memory();
int osx_get_time_mili();
int osx_get_width();
int osx_get_height();
void osx_start_thread(void (*func)(void*), void* data);
