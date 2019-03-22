#pragma once

//TODO: @REFACTOR file to just be memory_util


#ifdef _WIN32

#define W_ALIGN(x) __declspec( align (x) )
#define U_ALIGN(x) /*nothing*/
//This isn't specifically alignment.

#define alloca _alloca

#else

#define W_ALIGN(x) /*nothing*/
#define U_ALIGN(x) __attribute__(( aligned (x) ))

#endif
