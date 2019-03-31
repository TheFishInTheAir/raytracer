#ifdef _MEM_DEBUG
void* _debug_memcpy(void* dest, void* from, size_t size, int line, const char *func)
{
	printf("\n-");
	memcpy(dest, from, size);
	printf("- memcpy at %i, %s, %p[%li]\n\n", line, func, dest, size);
	fflush(stdout);
	return dest;
}
void* _debug_malloc(size_t size, int line, const char *func)
{
	printf("\n-");
	void *p = malloc(size);
	printf("- Allocation at %i, %s, %p[%li]\n\n", line, func, p, size);
	fflush(stdout);
	return p;
}

void _debug_free(void* ptr, int line, const char *func)
{
	printf("\n-");
	free(ptr);
	printf("- Free at %i, %s, %p\n\n", line, func, ptr);
	fflush(stdout);
}


#define malloc(X) _debug_malloc( X, __LINE__, __FUNCTION__)
#define free(X) _debug_free( X, __LINE__, __FUNCTION__)
#define memcpy(X, Y, Z) _debug_memcpy( X, Y, Z, __LINE__, __FUNCTION__)

#endif

#ifdef WIN32
#define DEBUG_BREAK __debugbreak
#define _FILE_SEP '\\'
#else
#define DEBUG_BREAK
#define _FILE_SEP '/'
#endif

#define __FILENAME__ (strrchr(__FILE__, _FILE_SEP) ? strrchr(__FILE__, _FILE_SEP) + 1 : __FILE__)


//TODO: replace all errors with this.
#define ASRT_CL(m)                                                                            \
    if(err!=CL_SUCCESS)                                                                       \
    {                                                                                         \
        fprintf(stderr, "ERROR: %s. (code: %i, line: %i, file:%s)\nPRESS ENTER TO EXIT\n", \
            m, err, __LINE__, __FILENAME__);                            \
        fflush(stderr);                                                 \
        while(1){char c; scanf("%c",&c); exit(1);}                      \
    }
//DEBUG_BREAK();                                                \
