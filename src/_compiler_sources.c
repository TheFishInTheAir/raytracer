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
//#define _MSC_VER //NOTE: this is necessary for OpenCL I know its not good
#define WIN32 // I don't want to fix all of my accidents right now.
#endif



//REMOVE FOR PRESENTATION
#define DEV_MODE



#ifdef WIN32
#include <win32.c>
//#else
//#include <osx.m>
#endif

//TODO: should put in a header
/*#ifdef WIN32
#define W_ALIGN(x) __declspec( align (x) )
#define U_ALIGN(x) /*nothing*//*
#else
#define W_ALIGN(x) /*nothing*//*
#define U_ALIGN(x) __attribute__ (( aligned (x) ))
#endif*/

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
