#include <math.h>
#include <stdlib.h>

#define MMX_IMPLEMENTATION
#include <vec.h>
#undef  MMX_IMPLEMENTATION

#ifdef WIN32
#include <win32.c>
#endif

#include <startup.c>
#include <geom.c>
#include <parallel.c>

/*
int main()
{
    printf("TEST\n");
    return 0;
}
*/
