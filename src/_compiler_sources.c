#include <math.h>
#include <stdlib.h>

#define MMX_IMPLEMENTATION
#include <vec.h>
#undef  MMX_IMPLEMENTATION
#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include <tinyobj_loader_c.h>
#undef TINYOBJ_LOADER_C_IMPLEMENTATION



#include <parson.c>

#define WIN32 // I guess CL doesn't add this macro...


#ifdef WIN32
#include <win32.c>
#endif

//TODO: should prob put in a header
#ifdef WIN32
#define W_ALIGN(x) __declspec( align (x) )
#define U_ALIGN(x) /*nothing*/
#else
#define W_ALIGN(x) /*nothing*/
#define U_ALIGN(x) __attribute__ ((aligned (x)));
#endif

//#define _MEM_DEBUG
#include <debug.c>

#include <os_abs.c>
#include <startup.c>
#include <scene.c>
#include <geom.c>
#include <loader.c>
#include <parallel.c>
#include <raytracer.c>
/*
int main()
{
    printf("TEST\n");
    return 0;
}
*/
