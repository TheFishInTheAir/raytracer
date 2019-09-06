#define CL_TARGET_OPENCL_VERSION 120

#include <math.h>
#include <stdlib.h>

#define MMX_IMPLEMENTATION
#include <vec.h>
#undef  MMX_IMPLEMENTATION
#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include <tinyobj_loader_c.h>
#undef TINYOBJ_LOADER_C_IMPLEMENTATION


#include <mongoose.c>
#include <parson.c>

#ifdef _WIN32
#define WIN32 // I don't want to fix all of my accidents right now.
#endif



//REMOVE FOR PRESENTATION
//#define DEV_MODE



#ifdef WIN32
#include <win32.c>
#endif
//NOTE: osx.m is compiled seperatly and then linked at the end.

//#define _MEM_DEBUG //Enable verbose memory allocation, movement and freeing

#include <CL/opencl.h>

#include <debug.c>

#include <os_abs.c>
#include <startup.c>
#include <scene.c>
#include <geom.c>
#include <loader.c>
#include <parallel.c>
#include <ui.c>
#include <irradiance_cache.c>
#include <raytracer.c>
#include <ss_raytracer.c>
#include <path_raytracer.c>
#include <spath_raytracer.c>
#include <kdtree.c>
