#include <geom.h>
#define DEBUG_PRINT_VEC3(n, v) printf(n ": (%f, %f, %f)\n", v[0], v[1], v[2])
#define EPSILON 0.0000001f

bool solve_quadratic(float *a, float *b, float *c, float *x0, float *x1)
{
    float discr = (*b) * (*b) - 4 * (*a) * (*c);

    if (discr < 0) return false;
    else if (discr == 0) {
        (*x0) = (*x1) = - 0.5 * (*b) / (*a);
    }
    else {
        float q = (*b > 0) ?
            -0.5 * (*b + sqrt(discr)) :
            -0.5 * (*b - sqrt(discr));
        *x0 = q / *a;
        *x1 = *c / q;
    }

    return true;
}

float* matvec_mul(mat4 m, vec4 v)
{
    float* out_float = (float*)malloc(sizeof(vec4));

    out_float[0] = m[0+0*4]*v[0] + m[0+1*4]*v[1] + m[0+2*4]*v[2] + m[0+3*4]*v[3];
    out_float[1] = m[1+0*4]*v[0] + m[1+1*4]*v[1] + m[1+2*4]*v[2] + m[1+3*4]*v[3];
    out_float[2] = m[2+0*4]*v[0] + m[2+1*4]*v[1] + m[2+2*4]*v[2] + m[2+3*4]*v[3];
    out_float[3] = m[3+0*4]*v[0] + m[3+1*4]*v[1] + m[3+2*4]*v[2] + m[3+3*4]*v[3];

    return out_float;
}

void swap_float(float *f1, float *f2)
{
    float temp = *f2;
    *f2 = *f1;
    *f1 = temp;
}


inline void AABB_divide(AABB source, uint8_t k, float b, AABB* left, AABB* right)
{
    vec3 new_min, new_max;
    memcpy(new_min, source.min, sizeof(vec3));
    memcpy(new_max, source.max, sizeof(vec3));

    float wrld_split = source.min[k] + (source.max[k] - source.min[k]) * b;
    new_min[k] = new_max[k] = wrld_split;

    memcpy(left->min,  source.min, sizeof(vec3));
    memcpy(left->max,  new_max,     sizeof(vec3));
    memcpy(right->min, new_min,     sizeof(vec3));
    memcpy(right->max, source.max, sizeof(vec3));
}


inline void AABB_divide_world(AABB source, uint8_t k, float world_b, AABB* left, AABB* right)
{
    vec3 new_min, new_max;
    memcpy(new_min, source.min, sizeof(vec3));
    memcpy(new_max, source.max, sizeof(vec3));

    new_min[k] = new_max[k] = world_b;

    memcpy(left->min,  source.min, sizeof(vec3));
    memcpy(left->max,  new_max,    sizeof(vec3));
    memcpy(right->min, new_min,    sizeof(vec3));
    memcpy(right->max, source.max, sizeof(vec3));
}


inline float AABB_surface_area(AABB source)
{
    vec3 diff;

    xv_sub(diff, source.max, source.min, 3);

    return (diff[0]*diff[1]*2 +
            diff[1]*diff[2]*2 +
            diff[0]*diff[2]*2);
}

inline void AABB_clip(AABB* result, AABB* target, AABB* container)
{
    memcpy(result,  target, sizeof(AABB));

    for (int i = 0; i < 3; i++)
    {
        if(result->min[i] < container->min[i])
            result->min[i] = container->min[i];
        if(result->max[i] > container->max[i])
            result->max[i] = container->max[i];
    }
}

inline void AABB_construct_from_triangle(AABB* result, ivec3* indices, vec3* vertices)
{
    for(int k = 0; k < 3; k++)
    {
        result->min[k] =  1000000;
        result->max[k] = -1000000;
    }

    for(int i = 0; i < 3; i++)
    {
        float* vertex = vertices[indices[i][0]];

        for(int k = 0; k < 3; k++)
        {
            if(vertex[k] < result->min[k])
                result->min[k] = vertex[k];

            if(vertex[k] > result->max[k])
                result->max[k] = vertex[k];
        }
    }
}

inline void AABB_construct_from_vertices(AABB* result, vec3* vertices,
                                          unsigned int num_vertices)
{
    for(int k = 0; k < 3; k++)
    {
        result->min[k] =  1000000;
        result->max[k] = -1000000;
    }
    for(int i = 0; i < num_vertices; i++)
    {
        for(int k = 0; k < 3; k++)
        {
            if(vertices[i][k] < result->min[k])
                result->min[k] = vertices[i][k];

            if(vertices[i][k] > result->max[k])
                result->max[k] = vertices[i][k];
        }
    }
}

inline bool AABB_is_planar(AABB* source, uint8_t k)
{
    if(source->max[k]-source->min[k] <= EPSILON)
        return true;
    return false;
}

inline float AABB_ilerp(AABB source, uint8_t k, float world_b)
{
    return (world_b - source.min[k]) / (source.max[k] - source.min[k]);
}

inline float does_collide_sphere(sphere s, ray r)
{
    float t0, t1; // solutions for t if the ray intersects


    vec3 L;
    xv_sub(L, r.orig, s.pos, 3);


    float a = 1.0f; //NOTE: we always normalize the direction vector.
    float b = xv3_dot(r.dir, L) * 2.0f;
    float c = xv3_dot(L, L) - (s.radius*s.radius); //NOTE: square can be optimized out.
    if (!solve_quadratic(&a, &b, &c, &t0, &t1)) return -1.0f;

    if (t0 > t1) swap_float(&t0, &t1);

    if (t0 < 0) {
        t0 = t1; // if t0 is negative, use t1 instead
        if (t0 < 0) return -1.0f; // both t0 and t1 are negative
    }

    return t0;
}

inline float does_collide_plane(plane p, ray r)
{
    float denom = xv3_dot(r.dir, p.norm);
    if (denom > 1e-6)
    {
        vec3 l;
        xv_sub(l, p.pos, r.orig, 3);
        float t = xv3_dot(l, p.norm) / denom;
        if (t >= 0)
            return -1.0;
        return t;
    }
    return -1.0;
}

ray generate_ray(int x, int y, int width, int height, float fov)
{
    ray r;

    //Simplified
    /* float ndc_x =((float)x+0.5)/width; */
    /* float ndc_y =((float)x+0.5)/height; */
    /* float screen_x = 2 ∗ ndc_x − 1; */
    /* float screen_y = 1 − 2 ∗ ndc_y; */
    /* float aspect_ratio = width/height; */
    /* float cam_x =(2∗screen_x−1) * tan(fov / 2 * M_PI / 180) ∗ aspect_ratio; */
    /* float cam_y = (1−2∗screen_y) * tan(fov / 2 * M_PI / 180); */

    float aspect_ratio = width / (float)height; // assuming width > height
    float cam_x = (2 * (((float)x + 0.5) / width) - 1) * tan(fov / 2 * M_PI / 180) * aspect_ratio;
    float cam_y = (1 - 2 * (((float)y + 0.5) / height)) * tan(fov / 2 * M_PI / 180);


    xv3_zero(r.orig);
    vec3 v1 = {cam_x, cam_y, -1};
    xv_sub(r.dir, v1, r.orig, 3);
    xv_normeq(r.dir, 3);

    return r;
}
