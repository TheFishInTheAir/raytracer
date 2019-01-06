#pragma once
#undef _WIN32
#ifdef _WIN32

#define W_ALIGN(x) __declspec( align (x) )
#define U_ALIGN(x) /*nothing*/

#else

#define W_ALIGN(x) /*nothing*/
#define U_ALIGN(x) __attribute__(( aligned (x) ))

#endif
