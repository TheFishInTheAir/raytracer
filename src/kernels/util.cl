
#define FOV 80.0f

#define vec2 float2
#define vec3 float3
#define vec4 float4

#define EPSILON 0.0000001f
#define FAR_PLANE 100000000

typedef float mat4[16];
typedef __global uint4* kd_44_matrix;



/********/
/* Util */
/********/


__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE   |
    CLK_FILTER_NEAREST;

typedef struct
{
    vec4 x;
    vec4 y;
    vec4 z;
    vec4 w;
} __attribute__((aligned (16))) t_mat4;

typedef struct kd_tree_collision_result
{
    unsigned int triangle_index;
    float t;
    float u;
    float v;
} kd_tree_collision_result;

void swap_float(float *f1, float *f2)
{
    float temp = *f2;
    *f2 = *f1;
    *f1 = temp;
}

vec4 matvec(float* m, vec4 v)
{
    return (vec4) (
        m[0+0*4]*v.x + m[1+0*4]*v.y + m[2+0*4]*v.z + m[3+0*4]*v.w,
        m[0+1*4]*v.x + m[1+1*4]*v.y + m[2+1*4]*v.z + m[3+1*4]*v.w,
        m[0+2*4]*v.x + m[1+2*4]*v.y + m[2+2*4]*v.z + m[3+2*4]*v.w,
        m[0+3*4]*v.x + m[1+3*4]*v.y + m[2+3*4]*v.z + m[3+3*4]*v.w );
}

unsigned int get_colour(vec4 col)
{
    unsigned int outCol = 0;

    col = clamp(col, 0.0f, 1.0f);

    outCol |= 0xff000000 & (unsigned int)(col.w*255)<<24;
    outCol |= 0x00ff0000 & (unsigned int)(col.x*255)<<16;
    outCol |= 0x0000ff00 & (unsigned int)(col.y*255)<<8;
    //outCol |= 0x000000ff & (unsigned int)(col.z*255);
    outCol |= 0x000000ff & (unsigned int)(col.z*255);

    /* outCol |= 0xff000000 & min((unsigned int)(col.w*255), (unsigned int)255)<<24; */
    /* outCol |= 0x00ff0000 & min((unsigned int)(col.x*255), (unsigned int)255)<<16; */
    /* outCol |= 0x0000ff00 & min((unsigned int)(col.y*255), (unsigned int)255)<<8; */
    /* outCol |= 0x000000ff & min((unsigned int)(col.z*255), (unsigned int)255); */
    return outCol;
}

static float get_random(unsigned int *seed0, unsigned int *seed1)
{
	/* hash the seeds using bitwise AND operations and bitshifts */
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);
	unsigned int ires = ((*seed0) << 16) + (*seed1);
	/* use union struct to convert int to float */
	union {
		float f;
		unsigned int ui;
	} res;
    //Maybe good, maybe not

	res.ui = (ires & 0x007fffff) | 0x40000000;  /* bitwise AND, bitwise OR */
	return (res.f - 2.0f) / 2.0f;
}

uint MWC64X(uint2 *state) //http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
{
    enum { A=4294883355U};
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);               // Pack the state back up
    return res;                       // Return the next result
}

vec3 reflect(vec3 incidentVec, vec3 normal)
{
    return incidentVec - 2.f * dot(incidentVec, normal) * normal;
}

__kernel void blit_float_to_output(
    __global unsigned int* out_tex,
    __global float* in_flts,
    const unsigned int width,
    const unsigned int height)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;
    out_tex[offset] = get_colour((vec4)(in_flts[offset]));
}

__kernel void blit_float3_to_output(
    __global unsigned int* out_tex,
    image2d_t in_flts,
    const unsigned int width,
    const unsigned int height)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;
    out_tex[offset] = get_colour(read_imagef(in_flts, sampler, (float2)(x, y)));
}


#ifdef DEBUG
//NOTE: this will be slow.
#define assert(x)                                                       \
    if (! (x))                                                          \
    {                                                                   \
        int i = 0;while(i++ < 100)printf("Assert(%s) failed in %s:%d\n", #x, __FUNCTION__, __LINE__); \
        return;                                                         \
    }
#else
#define assert(x)  //Nothing
#endif

void dbg_print_matrix(kd_44_matrix m)
{
    printf("[%2u %2u %2u %2u]\n" \
           "[%2u %2u %2u %2u]\n" \
           "[%2u %2u %2u %2u]\n" \
           "[%2u %2u %2u %2u]\n\n",
           m[0].x, m[0].y, m[0].z, m[0].w,
           m[1].x, m[1].y, m[1].z, m[1].w,
           m[2].x, m[2].y, m[2].z, m[2].w,
           m[3].x, m[3].y, m[3].z, m[3].w);
}
