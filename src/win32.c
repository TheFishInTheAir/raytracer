#include <win32.h>
#include <startup.h>
#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
const char CLASS_NAME[] = "Raytracer";


static win32_context* ctx;


os_abs init_win32_abs()
{
    os_abs abstraction;
    abstraction.start_func = &win32_start;
    abstraction.loop_start_func = &win32_loop;
    abstraction.update_func = &win32_update;
    abstraction.sleep_func = &win32_sleep;
    abstraction.get_bitmap_memory_func = &win32_get_bitmap_memory;
    abstraction.get_time_mili_func = &win32_get_time_mili;
    abstraction.get_width_func = &win32_get_width;
    abstraction.get_height_func = &win32_get_height;
    abstraction.start_thread_func = &win32_start_thread;
    return abstraction;
}

void* get_bitmap_memory()
{
    return ctx->bitmap_memory;
}

void win32_draw_meme()
{
    int width  = ctx->width;
    int height = ctx->height;

    int pitch = width*4;
    uint8_t* row = (uint8_t*)ctx->bitmap_memory;

    for(int y = 0; y < height; y++)
    {
        uint8_t* pixel = (uint8_t*)row;
        for(int x = 0; x < width; x++)
        {
            *pixel = sin(((float)x)/150)*255;
            ++pixel;

            *pixel = cos(((float)x)/10)*100;
            ++pixel;

            *pixel = cos(((float)y)/50)*255;
            ++pixel;

            *pixel = 0;
            ++pixel;
            /* ((char*)ctx->bitmap_memory)[(x+y*width)*4]   =  (y%2) ? 0xff : 0x00; */
            /* ((char*)ctx->bitmap_memory)[(x*4+y*width)+1] =  0x00; */
            /* ((char*)ctx->bitmap_memory)[(x*4+y*width)+2] =  (y%2) ? 0xff : 0x00; */
            /* ((char*)ctx->bitmap_memory)[(x*4+y*width)+3] =  0x00; */
        }
        row += pitch;
    }
}

void win32_sleep(int mili)
{
    Sleep(mili);
}

void win32_resize_dib_section(int width, int height)
{
    if(ctx->bitmap_memory)
        VirtualFree(ctx->bitmap_memory, 0, MEM_RELEASE);

    ctx->width = width;
    ctx->height = height;

    ctx->bitmap_info.bmiHeader.biSize          = sizeof(ctx->bitmap_info.bmiHeader);
    ctx->bitmap_info.bmiHeader.biWidth         = width;
    ctx->bitmap_info.bmiHeader.biHeight        = -height;
    ctx->bitmap_info.bmiHeader.biPlanes        = 1;
    ctx->bitmap_info.bmiHeader.biBitCount      = 32; //8 bits of paddingll
    ctx->bitmap_info.bmiHeader.biCompression   = BI_RGB;
    ctx->bitmap_info.bmiHeader.biSizeImage     = 0;
    ctx->bitmap_info.bmiHeader.biXPelsPerMeter = 0;
    ctx->bitmap_info.bmiHeader.biYPelsPerMeter = 0;
    ctx->bitmap_info.bmiHeader.biClrUsed       = 0;
    ctx->bitmap_info.bmiHeader.biClrImportant  = 0;

    //I could use BitBlit if it would increase spead.
    int bytes_per_pixel = 4;
    int bitmap_memory_size = (width*height)*bytes_per_pixel;
    ctx->bitmap_memory = VirtualAlloc(0, bitmap_memory_size, MEM_COMMIT, PAGE_READWRITE);

}

void win32_update_window(HDC device_context, HWND win, int width, int height)
{

    int window_height = height;//window_rect.bottom - window_rect.top;
    int window_width  = width;//window_rect.right - window_rect.left;


    //TODO: Replace with BitBlt this is way too slow... (we don't even need the scaling);
    StretchDIBits(device_context,
                  /* x, y, width, height, */
                  /* x, y, width, height, */
                  0, 0, ctx->width, ctx->height,
                  0, 0, window_width, window_height,

                  ctx->bitmap_memory,
                  &ctx->bitmap_info,
                  DIB_RGB_COLORS, SRCCOPY);
}


LRESULT CALLBACK WndProc(HWND win, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
    case WM_KEYDOWN:
        switch (wParam)
        {
        case VK_ESCAPE:
            loop_exit();
            ctx->shouldRun = false;
            break;

        case VK_SPACE:
            loop_pause();
            break;
        default:
            break;
        }
        break;
    case WM_SIZE:
    {
        RECT drawable_rect;
        GetClientRect(win, &drawable_rect);

        int height = drawable_rect.bottom - drawable_rect.top;
        int width  = drawable_rect.right - drawable_rect.left;
        win32_resize_dib_section(width, height);

        win32_draw_meme();
    } break;
    case WM_CLOSE:
        ctx->shouldRun = false;
        break;
    case WM_DESTROY:
        ctx->shouldRun = false;
        break;
    case WM_ACTIVATEAPP:
        OutputDebugStringA("WM_ACTIVATEAPP\n");
        break;
    case WM_PAINT:
    {
        PAINTSTRUCT paint;
        HDC device_context = BeginPaint(win, &paint);
        EndPaint(win, &paint);

        /*int x = paint.rcPaint.left;
        int y = paint.rcPaint.top;
        int height = paint.rcPaint.bottom - paint.rcPaint.top;
        int width  = paint.rcPaint.right - paint.rcPaint.left;*/
        //PatBlt(device_context, x, y, width, height, BLACKNESS);

        RECT drawable_rect;
        GetClientRect(win, &drawable_rect);

        int height = drawable_rect.bottom - drawable_rect.top;
        int width  = drawable_rect.right - drawable_rect.left;

        GetClientRect(win, &drawable_rect);
        win32_update_window(device_context,
                            win, width, height);

    } break;
    default:
        return DefWindowProc(win, msg, wParam, lParam);
    }
    return 0;
}



int _WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nCmdShow)
{

    ctx = (win32_context*) malloc(sizeof(win32_context));

    ctx->instance = hInstance;
    ctx->nCmdShow = nCmdShow;
    ctx->wc.cbSize        = sizeof(WNDCLASSEX);
    ctx->wc.style         = CS_OWNDC|CS_HREDRAW|CS_VREDRAW;
    ctx->wc.lpfnWndProc   = WndProc;
    ctx->wc.cbClsExtra    = 0;
    ctx->wc.cbWndExtra    = 0;
    ctx->wc.hInstance     = hInstance;
    ctx->wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    ctx->wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    ctx->wc.hbrBackground = 0;//(HBRUSH)(COLOR_WINDOW+1);
    ctx->wc.lpszMenuName  = NULL;
    ctx->wc.lpszClassName = CLASS_NAME;
    ctx->wc.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);

    if(!SetPriorityClass(
           GetCurrentProcess(),
           HIGH_PRIORITY_CLASS
           ))
    {
        printf("FUCKKKK!!!\n");
    }

    startup();

    return 0;
}

int main()
{
    //printf("JANKY WINMAIN OVERRIDE\n");
    return _WinMain(GetModuleHandle(NULL), NULL, GetCommandLineA(), SW_SHOWNORMAL);
}

//Should Block the Win32 Update Loop.
#define WIN32_SHOULD_BLOCK_LOOP

void win32_loop()	
{
    printf("Starting WIN32 Window Loop\n");
    MSG msg;
    ctx->shouldRun = true;
    while(ctx->shouldRun)
    {
#ifdef WIN32_SHOULD_BLOCK_LOOP


        if(GetMessage(&msg, 0, 0, 0) > 0)
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

#else
        while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
        {
            if(msg.message == WM_QUIT)
            {
                ctx->shouldRun = false;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
#endif
        //win32_draw_meme();
        //win32_update_window();
    }
}


void create_win32_window()
{
    printf("Creating WIN32 Window\n");

    ctx->win = CreateWindowEx(
        0,
        CLASS_NAME,
        CLASS_NAME,
        /* WS_OVERLAPPEDWINDOW, */
        (WS_POPUP| WS_SYSMENU | WS_MAXIMIZEBOX | WS_MINIMIZEBOX),
        CW_USEDEFAULT, CW_USEDEFAULT, 1920, 1080,
        NULL, NULL, ctx->instance, NULL);

    if(ctx->win == NULL)
    {
        MessageBox(NULL, "Window Creation Failed!", "Error!",
                   MB_ICONEXCLAMATION | MB_OK);
        return;
    }

    ShowWindow(ctx->win, ctx->nCmdShow);
    UpdateWindow(ctx->win);

}


//NOTE: Should the start func start the loop
//#define WIN32_SHOULD_START_LOOP_ON_START
void win32_start()
{
    if(!RegisterClassEx(&ctx->wc))
    {
        MessageBox(NULL, "Window Registration Failed!", "Error!",
                   MB_ICONEXCLAMATION | MB_OK);
        return;
    }
    create_win32_window();
#ifdef WIN32_SHOULD_START_LOOP_ON_START
    win32_loop();
#endif

}

int win32_get_time_mili()
{
    SYSTEMTIME st;
    GetSystemTime(&st);
    return (int) st.wMilliseconds+(st.wSecond*1000)+(st.wMinute*1000*60);
}

void win32_update()
{
    //RECT win_rect;
    //GetClientRect(ctx->win, &win_rect);
    HDC dc = GetDC(ctx->win);
    win32_update_window(dc, ctx->win, ctx->width, ctx->height);
    ReleaseDC(ctx->win, dc);

}


int win32_get_width()
{
    return ctx->width;
}

int win32_get_height()
{
    return ctx->height;
}

void* win32_get_bitmap_memory()
{
    return ctx->bitmap_memory;
}


typedef struct
{
    void* data;
    void (*func)(void*);
} thread_func_meta;

DWORD WINAPI thread_func(void* data)
{
    if(!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST))
    {
        DWORD dwError;
        dwError = GetLastError();
        printf(TEXT("Failed to change thread priority (%d)\n"), dwError);
    }

    thread_func_meta* meta = (thread_func_meta*) data;
    (meta->func)(meta->data); //confusing syntax: call the passed function with the passed data
    free(meta);
    return 0;
}

void win32_start_thread(void (*func)(void*), void* data)
{
    thread_func_meta* meta = (thread_func_meta*) malloc(sizeof(thread_func_meta));
    meta->data = data;
    meta->func = func;
    HANDLE t = CreateThread(NULL, 0, thread_func, meta, 0, NULL);
    //if(SetThreadPriority(t, THREAD_PRIORITY_HIGHEST)==0)
    //    assert(false);

}
