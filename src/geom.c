#include <geom.h>
#define DEBUG_PRINT_VEC3(n, v) printf(n ": (%f, %f, %f)\n", v[0], v[1], v[2])


//I Don't Understand this TODO: understand. UPDATE: I sort of understand it now
inline bool solve_quadratic(float *a, float *b, float *c, float *x0, float *x1)
{
    float discr = (*b) * (*b) - 4 * (*a) * (*c);
    // printf("test: %f    a:%f, b:%f, c:%f\n", discr, *a, *b, *c);
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

void swap_float(float *f1, float *f2)
{
    float temp = *f2;
    *f2 = *f1;
    *f1 = temp;
}

inline float does_collide_sphere(sphere s, ray r)
{
    float t0, t1; // solutions for t if the ray intersects

    // analytic solution
    vec3 L;
    xv_sub(L, r.orig, s.pos, 3);
    //DEBUG_PRINT_VEC3("DIR", r.dir);
    //DEBUG_PRINT_VEC3("SPOS", s.pos);
    //DEBUG_PRINT_VEC3("L", L);

    //float a = xv3_dot(r.dir, r.dir);
    float a = 1.0f; //NOTE: we always normalize the direction vector.
    float b = xv3_dot(r.dir, L) * 2.0f;
    float c = xv3_dot(L, L) - (s.radius*s.radius); //NOTE: you can optimize out the square.
    if (!solve_quadratic(&a, &b, &c, &t0, &t1)) return -1.0f;

    if (t0 > t1) swap_float(&t0, &t1);

    if (t0 < 0) {
        t0 = t1; // if t0 is negative, let's use t1 instead
        if (t0 < 0) return -1.0f; // both t0 and t1 are negative
    }


    return t0;

    //NOTE: OLD
    /* vec3 l; */
    /* xv_sub(l, r.orig, s.pos,  3); */

    /* float tca = xv_dot(l, r.dir, 3); //NOTE: we assue r.dir is normalized */
    /* if (tca < 0) return -1.0f; */
    /* float d2 = xv3_dot(l, l) - tca * tca; */
    /* if (d2 > s.radius*s.radius) return -1.0f; */
    /* return true; */
}

inline float does_collide_plane(plane p, ray r)
{
    float denom = xv_dot3(r.dir, p.norm);
    if (denom > 1e-6)
    {
        vec3 l;
        xv_sub(l, p.pos, r.orig, 3);
        float t = xv_dot3(l, p.norm) / denom;
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

//    vec3 new_orig = {0,0,0};
    //  r.orig = new_orig;
    xv3_zero(r.orig);
    vec3 v1 = {cam_x, cam_y, -1};
    xv_sub(r.dir, v1, r.orig, 3);
    xv_normeq(r.dir, 3);

    return r;
}
