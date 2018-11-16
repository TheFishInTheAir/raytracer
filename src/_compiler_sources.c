#include <math.h>
#include <stdlib.h>

#define MMX_IMPLEMENTATION
#include <vec.h>
#undef  MMX_IMPLEMENTATION

#define WIN32 // I guess CL doesn't add this macro...


#ifdef WIN32
#include <win32.c>
#endif

#include <os_abs.c>
#include <startup.c>
#include <geom.c>
#include <parallel.c>
#include <raytracer.c>

/*
int main()
{
    printf("TEST\n");
    return 0;
}
*/
