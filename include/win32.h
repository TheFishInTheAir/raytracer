#pragma once
#include <windows.h>
#include <stdbool.h>
#include <os_abs.h>
//TODO: add v-table for this

typedef struct
{
    HINSTANCE instance;
    int       nCmdShow;
    WNDCLASSEX wc;
    HWND     win;

    int width, height;

    BITMAPINFO bitmap_info;
    void*      bitmap_memory;

    // HDC        render_device_context;

    bool       shouldRun;
    //Bitbuffer
} win32_context;


os_abs init_win32_abs();

void win32_start_thread(void (*func)(void*), void* data);

//void create_win32_window();
void win32_start();
void win32_loop();

void win32_update();

void win32_sleep(int);

void* win32_get_bitmap_memory();

int win32_get_time_mili();

int win32_get_width();
int win32_get_height();
