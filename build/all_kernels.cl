#define FOV 80.0f

#define vec3 float3
#define vec4 float4

#define EPSILON 0.0000001f
#define FAR_PLANE 100000000

typedef float mat4[16];



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
    outCol |= 0x000000ff & (unsigned int)(col.z*255);


    /* outCol |= 0xff000000 & min((unsigned int)(col.w*255), (unsigned int)255)<<24; */
    /* outCol |= 0x00ff0000 & min((unsigned int)(col.x*255), (unsigned int)255)<<16; */
    /* outCol |= 0x0000ff00 & min((unsigned int)(col.y*255), (unsigned int)255)<<8; */
    /* outCol |= 0x000000ff & min((unsigned int)(col.z*255), (unsigned int)255); */
    return outCol;
    //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.

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

	res.ui = (ires & 0x007fffff) | 0x40000000;  /* bitwise AND, bitwise OR */
	return (res.f - 2.0f) / 2.0f;
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
typedef union
{
    float arr[4];
    vec3  vec;
} hack_vec3;

//NOTE: from Graphics Gems 1990 (Andrew Woo and John Amantides)
#define NUMDIM	3
#define RIGHT	0
#define LEFT	1
#define MIDDLE	2
bool hitBoundingBox(hack_vec3 minB, hack_vec3 maxB,
                    hack_vec3 origin, hack_vec3 dir, hack_vec3 coord)
{
	bool inside = true;
	char quadrant[NUMDIM];
	register int i;
	int whichPlane;
	float maxT[NUMDIM];
	float candidatePlane[NUMDIM];

	/* Find candidate planes; this loop can be avoided if
   	rays cast all from the eye(assume perpsective view) */
	for (i=0; i<NUMDIM; i++)
		if(origin.arr[i] < minB.arr[i]) {
			quadrant[i] = LEFT;
			candidatePlane[i] = minB.arr[i];
			inside = false;
		}else if (origin.arr[i] > maxB.arr[i]) {
			quadrant[i] = RIGHT;
			candidatePlane[i] = maxB.arr[i];
			inside = false;
		}else	{
			quadrant[i] = MIDDLE;
		}

	/* Ray origin inside bounding box */
	if(inside)	{
		coord = origin;
		return true; //should be true
	}


	/* Calculate T distances to candidate planes */
	for (i = 0; i < NUMDIM; i++)
		if (quadrant[i] != MIDDLE && dir.arr[i] !=0.)
			maxT[i] = (candidatePlane[i]-origin.arr[i]) / dir.arr[i];
		else
			maxT[i] = -1.;

	/* Get largest of the maxT's for final choice of intersection */
	whichPlane = 0;
	for (i = 1; i < NUMDIM; i++)
		if (maxT[whichPlane] < maxT[i])
			whichPlane = i;

	/* Check final candidate actually inside box */
	if (maxT[whichPlane] < 0.) return false;
	for (i = 0; i < NUMDIM; i++)
		if (whichPlane != i) {
			coord.arr[i] = origin.arr[i] + maxT[whichPlane] * dir.arr[i];
			if (coord.arr[i] < minB.arr[i] || coord.arr[i] > maxB.arr[i])
				return false;
		} else {
			coord.arr[i] = candidatePlane[i];
		}
	return true;				/* ray hits box */
}


/************/
/* Material */
/************/
typedef struct //16 bytes
{
    vec3 colour;

    float reflectivity;
} __attribute__ ((aligned (16))) material;

//TODO: refactor var names
/*material get_material(__global float* buf, int offset) //NOTE: offset is index (woule be a better name)
{
    int real_offset = offset*(4);

    material m;

    m.reflectivity = buf[0 + real_offset];
    m.colour.x     = buf[1 + real_offset];
    m.colour.y     = buf[2 + real_offset];
    m.colour.z     = buf[3 + real_offset];

    return m;
}*/

/*******/
/* Ray */
/*******/
typedef struct
{
    vec3 orig;
    vec3 dir;
} ray;

//OTHER THING
typedef struct
{
    bool did_hit;
    vec3 normal;
    vec3 point;
    float dist;
    material mat;
    //TODO: Add material
} collision_result;



/**********************/
/*                    */
/*     Primitives     */
/*                    */
/**********************/

/************/
/* Triangle */
/************/

//Moller-Trumbore
//t u v = x y z
bool does_collide_triangle(vec3 tri[3], vec3* hit_coords, ray r)
{
    vec3 ab = tri[1] - tri[0];
    vec3 ac = tri[2] - tri[0];

    vec3 pvec = cross(r.dir, ac); //Triple product
    float det = dot(ab, pvec);

    if (det < EPSILON) // Behind or close to parallel. NOTE: TEMP FABS
        return false;

    float invDet = 1.f / det;
    vec3 tvec = r.orig - tri[0];

    //u
    hit_coords->y = dot(tvec, pvec) * invDet;
    if(hit_coords->y < 0 || hit_coords->y > 1)
        return false;

    //v
    vec3 qvec = cross(tvec, ab);
    hit_coords->z = dot(r.dir, qvec) * invDet;
    if (hit_coords->z < 0 || hit_coords->y + hit_coords->z > 1)
        return false;

    //t
    hit_coords->x = dot(ac, qvec) * invDet;

    return true; //goose

}


/**********/
/* Sphere */
/**********/
typedef struct //32 bytes (one word)
{
    vec3 pos;
    //4 bytes padding
    float radius;
    int material_index;
    //8 bytes padding
} __attribute__ ((aligned (16))) sphere;

bool does_collide_sphere(sphere s, ray r, float *dist)
{
    float t0, t1; // solutions for t if the ray intersects

    // analytic solution
    vec3 L = s.pos- r.orig;
    float b = dot(r.dir, L) ;//* 2.0f;
    float c = dot(L, L) - (s.radius*s.radius); //NOTE: you can optimize out the square.

    float disc = b * b - c/**a*/; /* discriminant of quadratic formula */

    /* solve for t (distance to hitpoint along ray) */
    float t = false;

    if (disc < 0.0f) return false;
    else t = b - sqrt(disc);

    if (t < 0.0f)
    {
        t = b + sqrt(disc);
        if (t < 0.0f) return false;
    }
    *dist = t;
    return true;
}



/*********/
/* Plane */
/*********/

typedef struct plane
{
    vec3 pos;
    vec3 normal;

    int material_index;
} __attribute__ ((aligned (16))) plane;


bool does_collide_plane(plane p, ray r, float *dist)
{
    float denom = dot(r.dir, p.normal);
    if (denom < EPSILON) //Counter intuitive.
    {
        vec3 l = p.pos - r.orig;
        float t = dot(l, p.normal) / denom;
        if (t >= 0)
        {
            *dist = t;
            return true;
        }

    }
    return false;
}


/********************/
/*                  */
/*      Meshes      */
/*                  */
/********************/

typedef struct
{

    mat4 model;

    vec3 max;
    vec3 min;

    int index_offset;
    int num_indices;


    int material_index;
} __attribute__((aligned (32))) mesh; //TODO: align with cpu NOTE: I don't think we need 32

bool does_collide_with_mesh(mesh collider, ray r, vec3* normal, float* dist,
                            const __global int* indices,
                            const __global vec3* vertices,
                            const __global vec3* normals)
{
    //TODO: k-d trees
    *dist = FAR_PLANE;
    float min_t = FAR_PLANE;
    vec3 hit_coord; //NOTE: currently unused
    ray r2 = r;
    if(!hitBoundingBox((hack_vec3)collider.min, (hack_vec3)collider.max,
                       (hack_vec3)r.orig, (hack_vec3) r.dir, (hack_vec3) hit_coord))
        return false;

    //return false;
    for(int i = 0; i < collider.num_indices/3; i++) // each ivec3
    {
        vec3 tri[3]; //TODO: optmimze

        //get vertex (first element of each index)
        int idx_0 = indices[(i*3+collider.index_offset+0)*3]; //TODO: add offset
        int idx_1 = indices[(i*3+collider.index_offset+1)*3]; //
        int idx_2 = indices[(i*3+collider.index_offset+2)*3]; //

        tri[0] = vertices[idx_0];
        tri[1] = vertices[idx_1];
        tri[2] = vertices[idx_2];

        /*printf("%i/%i : (%.2f %.2f %.2f) (%.2f %.2f %.2f) (%.2f %.2f %.2f)\n", i, collider.num_indices/3,
               tri[0].x, tri[0].y, tri[0].z,
               tri[1].x, tri[1].y, tri[1].z,
               tri[2].x, tri[2].y, tri[2].z);*/


        vec3 bc_hit_coords; //t u v = x y z
        if(does_collide_triangle(tri, &bc_hit_coords, r))
        {
            //printf("NUT_0!! %f\n", bc_hit_coords.x);
            if(bc_hit_coords.x<min_t && bc_hit_coords.x>0)
            {

                min_t = bc_hit_coords.x; //t (distance along direction)

                int nidx_0 = indices[(i*3+collider.index_offset+0)*3+1]; //TODO: add offset
                int nidx_1 = indices[(i*3+collider.index_offset+1)*3+1]; //
                int nidx_2 = indices[(i*3+collider.index_offset+2)*3+1]; //

                vec3 anorm = normals[nidx_0]*(1-bc_hit_coords.y-bc_hit_coords.z); //w
                vec3 bnorm = normals[nidx_1]*bc_hit_coords.y; //u
                vec3 cnorm = normals[nidx_2]*bc_hit_coords.z; //v



                *normal = anorm+bnorm+cnorm;
                /*printf("TEST: %f %f %f: %f\n",
                       bc_hit_coords.y, bc_hit_coords.z,
                       1-(bc_hit_coords.y+bc_hit_coords.z),
                       bc_hit_coords.y+bc_hit_coords.z+1-(bc_hit_coords.y+bc_hit_coords.z)
                       );*/
                //printf("TEST: %f\n", fabs((normals[nidx_0]).x)+fabs((normals[nidx_0]).y)+fabs((normals[nidx_0]).z));

            }
        }

    }


    *dist = min_t;
    //if(r.dir.z>0&&min_t==FAR_PLANE)
    //    printf("fuck but good\n");
    return min_t != FAR_PLANE;

}



/************************/
/* High Level Collision */
/************************/


bool collide_meshes(ray r, collision_result* result,
                    const __global material* material_buffer,
                    const __global mesh* meshes,
                    const __global int*  indices,
                    const __global vec3* vertices,
                    const __global vec3* normals)
{

    float dist = FAR_PLANE;
    result->did_hit = false;
    result->dist = FAR_PLANE;

    for(int i = 0; i < SCENE_NUM_MESHES; i++)
    {
        mesh current_mesh = meshes[i];
        float local_dist = FAR_PLANE;
        vec3 normal;
        if(does_collide_with_mesh(current_mesh, r, &normal, &local_dist,
                                  indices, vertices, normals))
        {

            if(local_dist<dist)
            {
                dist = local_dist;
                result->dist = dist;
                result->normal = normal;
                result->point = (r.dir*dist)+r.orig;
                result->mat = material_buffer[current_mesh.material_index];
                result->did_hit = true;
            }
        }
    }
    return result->did_hit;
}

bool collide_primitives(ray r, collision_result* result,
                        const __global material* material_buffer,
                        const __global sphere* spheres,
                        const __global plane* planes)
{

    float dist = FAR_PLANE;
    result->did_hit = false;
    result->dist = FAR_PLANE;
    for(int i = 0; i < SCENE_NUM_SPHERES; i++)
    {
        sphere current_sphere = spheres[i];//get_sphere(spheres, i);
        float local_dist = FAR_PLANE;
        if(does_collide_sphere(current_sphere, r, &local_dist))
        {
            if(local_dist<dist)
            {
                dist = local_dist;
                result->did_hit = true;
                result->dist = dist;
                result->point  = r.dir*dist+r.orig;
                result->normal = normalize(result->point - current_sphere.pos);
                result->mat = material_buffer[current_sphere.material_index];
            }
        }
    }

    for(int i = 0; i < SCENE_NUM_PLANES; i++)
    {
        plane current_plane = planes[i];//get_plane(planes, i);
        float local_dist =  FAR_PLANE;
        if(does_collide_plane(current_plane, r, &local_dist))
        {
            if(local_dist<dist)
            {
                dist = local_dist;
                result->did_hit = true;
                result->dist   = dist;
                result->point  = r.dir*dist+r.orig;
                result->normal = current_plane.normal;
                result->mat = material_buffer[current_plane.material_index];
            }
        }
    }

    return dist != FAR_PLANE;
}

bool collide_all(ray r, collision_result* result,
                 const __global material* material_buffer,
                 const __global sphere* spheres,
                 const __global plane* planes,
                 const __global mesh* meshes,
                 const __global int*  indices,
                 const __global vec3* vertices,
                 const __global vec3* normals)
{
    float dist = FAR_PLANE;
    if(collide_primitives(r, result, material_buffer, spheres, planes))
        dist = result->dist;

    collision_result m_result;
    if(collide_meshes(r, &m_result, material_buffer, meshes, indices, vertices, normals))
        if(m_result.dist < dist)
            *result = m_result;

    return result->did_hit;
}
/**********************/
/* Irradiance Caching */
/**********************/


/* typedef struct */
/* { */
/*     vec3 pos; */
/*     size_t left, right; //pointers within buffer */

/*     //Irradiance cache */
/*     vec3 irradiance; */
/*     vec3 normal; */
/*     float angle_factor; */

/* } __attribute__((aligned (16)))  ic_kd_node; */

__kernel void ic_screen_textures(
    __write_only image2d_t pos_tex,
    __write_only image2d_t nrm_tex,
    const unsigned int width,
    const unsigned int height,
    const __global float* ray_buffer,
    const vec4 pos,
    const __global material* material_buffer,
    const __global sphere* spheres,
    const __global plane* planes,
    const __global mesh* meshes,
    const __global int* indices,
    const __global vec3* vertices,
    const __global vec3* normals)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;
    int ray_offset = offset*3;

    ray r;
    r.orig = pos.xyz; //NOTE: slow unaligned memory access.
    r.dir.x = ray_buffer[ray_offset];
    r.dir.y = ray_buffer[ray_offset+1];
    r.dir.z = ray_buffer[ray_offset+2];

    collision_result result;
    if(!collide_all(r, &result, material_buffer, spheres, planes, meshes, indices,
                    vertices, normals))
    {
        write_imagef(pos_tex, (int2)(x,y), (vec4)(0));
        write_imagef(nrm_tex, (int2)(x,y), (vec4)(0));
        return;
    }

    write_imagef(pos_tex, (int2)(x,y), (vec4)(result.point,0)); //Maybe ???
    write_imagef(nrm_tex, (int2)(x,y), (vec4)(result.normal,0));

    /* pos_tex[offset] = (vec4)(result.point,0); */
    /* nrm_tex[offset] = (vec4)(result.normal,0); */
}



__kernel void generate_discontinuity(
    image2d_t pos_tex,
    image2d_t nrm_tex,
    __global float* out_tex,
    const float k,
    const float intensity,
    const unsigned int width,
    const unsigned int height)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;

    //NOTE: this is fine for edges because the sampler is clamped

    //Positions
    vec4 pm = read_imagef(pos_tex, sampler, (int2)(x,y));
    vec4 pu = read_imagef(pos_tex, sampler, (int2)(x,y+1));
    vec4 pd = read_imagef(pos_tex, sampler, (int2)(x,y-1));
    vec4 pr = read_imagef(pos_tex, sampler, (int2)(x+1,y));
    vec4 pl = read_imagef(pos_tex, sampler, (int2)(x-1,y));

    //NOTE: slow doing this many distance calculations
    float posDiff = max(distance(pu,pm),
                        max(distance(pd,pm),
                            max(distance(pr,pm),
                                distance(pl,pm))));
    posDiff = clamp(posDiff, 0.f, 1.f);
    posDiff *= intensity;

    //Normals
    vec4 nm = read_imagef(nrm_tex, sampler, (int2)(x,y));

    vec4 nu = read_imagef(nrm_tex, sampler, (int2)(x,y+1));
    vec4 nd = read_imagef(nrm_tex, sampler, (int2)(x,y-1));
    vec4 nr = read_imagef(nrm_tex, sampler, (int2)(x+1,y));
    vec4 nl = read_imagef(nrm_tex, sampler, (int2)(x-1,y));
    //NOTE: slow doing this many distance calculations
    float nrmDiff = max(distance(nu,nm),
                        max(distance(nd,nm),
                            max(distance(nr,nm),
                                distance(nl,nm))));
    nrmDiff = clamp(nrmDiff, 0.f, 1.f);
    nrmDiff *= intensity;

    out_tex[offset] = k*nrmDiff+posDiff;
}

__kernel void float_average(
    __global float* in_tex,
    __global float* out_tex,
    const unsigned int width,
    const unsigned int height,
    const int total)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;

    out_tex[offset] += in_tex[offset]/(float)total;

}


__kernel void mip_single_upsample( //nearest neighbour upsample.
    __global float* in_tex,
    __global float* out_tex,
    const unsigned int width, //Of upsampled
    const unsigned int height)//Of upsampled
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;

    out_tex[offset] = in_tex[(x+y*width)/2]; //truncated
}

__kernel void mip_upsample( //nearest neighbour upsample.
    image2d_t in_tex,
    __write_only image2d_t out_tex, //NOTE: not having __write_only caused it to crash without err
    const unsigned int width, //Of upsampled
    const unsigned int height)//Of upsampled
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;

    write_imagef(out_tex, (int2)(x,y),
                 read_imagef(in_tex, sampler, (float2)((float)x/2.f, (float)y/2.f)));
}

__kernel void mip_upsample_scaled( //nearest neighbour upsample.
    image2d_t in_tex,
    __write_only image2d_t out_tex,
    const int s,
    const unsigned int width, //Of upsampled
    const unsigned int height)//Of upsampled
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    float factor = pow(2.f, (float)s);
    write_imagef(out_tex, (int2)(x,y),
                 read_imagef(in_tex, sampler, (float2)((float)x/factor, (float)y/factor)));
}
__kernel void mip_single_upsample_scaled( //nearest neighbour upsample.
    __global float* in_tex,
    __global float* out_tex,
    const unsigned int s,
    const unsigned int width, //Of upsampled
    const unsigned int height)//Of upsampled
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int factor = (int) pow(2.f, (float)s);
    int offset = x+y*width;
    int fwidth = width/factor;
    int fheight = height/factor;

    out_tex[offset] = in_tex[(x/factor)+(y/factor)*(width/factor)]; //truncated
}

//NOTE: not used
__kernel void mip_reduce( //not the best
    image2d_t in_tex,
    __write_only image2d_t out_tex,
    const unsigned int width, //Of reduced
    const unsigned int height)//Of reduced
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;



    vec4 p00 = read_imagef(in_tex, sampler, (int2)(x*2,   y*2  ));

    vec4 p01 = read_imagef(in_tex, sampler, (int2)(x*2+1, y*2  ));

    vec4 p10 = read_imagef(in_tex, sampler, (int2)(x*2,   y*2+1));

    vec4 p11 = read_imagef(in_tex, sampler, (int2)(x*2+1, y*2+1));

    write_imagef(out_tex, (int2)(x,y), p00+p01+p10+p11/4.f);
}

vec4 shade(collision_result result) //NOTE: Temp shitty phong
{
    const vec3 light_pos = (vec3)(2,5,-1);
    vec3 nspace_light_dir = normalize(light_pos-result.point);
    vec4 test_lighting = (vec4) (clamp((float)dot(result.normal, nspace_light_dir), 0.0f, 1.0f));
    test_lighting *= (vec4)(result.mat.colour, 1.0f);
    return test_lighting;
}


__kernel void cast_ray_test(
    __global unsigned int* out_tex,
    const __global float* ray_buffer,
    const __global material* material_buffer,
    const __global sphere* spheres,
    const __global plane* planes,
//Mesh
    const __global mesh* meshes,
    const __global int* indices,
    const __global vec3* vertices,
    const __global vec3* normals,
    /* const __global vec2* texcoords, */
    /* , */


    const unsigned int width,
    const unsigned int height,
    const vec4 pos)
{
    const vec4 sky = (vec4) (0.2, 0.8, 0.5, 0);
    //return;
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;
    int ray_offset = offset*3;


    ray r;
    r.orig = pos.xyz; //NOTE: unoptimized unaligned memory access.
    r.dir.x = ray_buffer[ray_offset];
    r.dir.y = ray_buffer[ray_offset+1];
    r.dir.z = ray_buffer[ray_offset+2];

    //r.dir  = (vec3)(0,0,-1);

    //out_tex[x+y*width] = get_colour_signed((vec4)(r.dir,0));
    //out_tex[x+y*width] = get_colour_signed((vec4)(1,1,0,0));
    collision_result result;
    if(!collide_all(r, &result, material_buffer, spheres, planes, meshes, indices,
                   vertices, normals))
    {
        out_tex[x+y*width] = get_colour( sky );
        return;
    }
    vec4 colour = shade(result);


    #define NUM_REFLECTIONS 2
    ray rays[NUM_REFLECTIONS];
    collision_result results[NUM_REFLECTIONS];
    vec4 colours[NUM_REFLECTIONS];
    int early_exit_num = NUM_REFLECTIONS;
    for(int i = 0; i < NUM_REFLECTIONS; i++)
    {
        if(i==0)
        {
            rays[i].orig = result.point + result.normal * 0.0001f; //NOTE: BIAS
            rays[i].dir  = reflect(r.dir, result.normal);//reflect(r.dir, result.norm);
        }
        else
        {
            rays[i].orig = results[i-1].point + results[i-1].normal * 0.0001f; //NOTE: BIAS
            rays[i].dir  = reflect(rays[i-1].dir, results[i-1].normal);//reflect(r.dir, result.norm);
        }
        if(collide_all(rays[i], results+i, material_buffer, spheres,
                       planes, meshes, indices, vertices, normals))
        {
            colours[i] = shade(results[i]);
        }
        else
        {
            colours[i] = sky;
            early_exit_num = i;
            break;
        }
    }
    for(int i = early_exit_num-1; i > -1; i--)
    {
        //if(i==0)
            //{
        if(i==NUM_REFLECTIONS-1)
            colours[i] = mix(colours[i], sky, results[i].mat.reflectivity);

        else
            colours[i] = mix(colours[i], colours[i+1], results[i].mat.reflectivity);


            //}
            //else
            //{

            //}
    }

    colour = mix(colour, colours[0],  result.mat.reflectivity);

    out_tex[x+y*width] = get_colour( colour );

    //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.

}


//NOTE: it might be faster to make the ray buffer a multiple of 4 just to fit word size...
__kernel void generate_rays(
    __global float* out_tex,
    const unsigned int width,
    const unsigned int height,
    const t_mat4 wcm)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width) * 3;

    ray r;

    float aspect_ratio = width / (float)height; // assuming width > height
    float cam_x = (2 * (((float)x + 0.5) / width) - 1) * tan(FOV / 2 * M_PI / 180) * aspect_ratio;
    float cam_y = (1 - 2 * (((float)y + 0.5) / height)) * tan(FOV / 2 * M_PI / 180);

    //r.orig = matvec((float*)&wcm, (vec4)(0.0, 0.0, 0.0, 1.0)).xyz;
    //r.dir  = matvec((float*)&wcm, (vec4)(cam_x, cam_y, -1.0f, 1)).xyz - r.orig;

    r.orig = (vec3)(0, 0, 0);
    r.dir  = (vec3)(cam_x, cam_y, -1.0f) - r.orig;

    r.dir = normalize(r.dir);

    out_tex[offset]   = r.dir.x;
    out_tex[offset+1] = r.dir.y;
    out_tex[offset+2] = r.dir.z;
}

vec3 uniformSampleHemisphere(const float r1, const float r2)
{
    // cos(theta) = r1 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
    float sinTheta = sqrt(1 - r1 * r1);
    float phi = 2 * M_PI * r2;
    float x = sinTheta * cos(phi);
    float z = sinTheta * sin(phi);
    return (vec3)(x, r1, z);
}
vec3 cosineSampleHemisphere(float u1, float u2, vec3 normal)
{
    const float r = sqrt(u1);
    const float theta = 2 * M_PI * u2;

    vec3 w = normal;
    vec3 axis = fabs(w.x) > 0.1f ? (vec3)(0.0f, 1.0f, 0.0f) : (vec3)(1.0f, 0.0f, 0.0f);
    vec3 u = normalize(cross(axis, w));
    vec3 v = cross(w, u);

    /* use the coordinte frame and random numbers to compute the next ray direction */
    return normalize(u * cos(theta)*r + v*sin(theta)*r + w*sqrt(1.0f - u1));

}

#define NUM_BOUNCES 4
#define NUM_SAMPLES 64
__kernel void path_trace(
    __global vec4* out_tex,
    const __global float* ray_buffer,
    const __global material* material_buffer,
    const __global sphere* spheres,
    const __global plane* planes,
//Mesh
    const __global mesh* meshes,
    const __global int* indices,
    const __global vec3* vertices,
    const __global vec3* normals,
    /* const __global vec2* texcoords, */
    /* , */
    const unsigned int width,
    const unsigned int height,
    const vec4 pos,
    unsigned int magic)
{
    const vec4 sky = (vec4) (0.16, 0.2, 0.2, 0);
    //return;
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;
    int ray_offset = offset*3;

    ray r;
    r.orig = pos.xyz;
    r.dir.x = ray_buffer[ray_offset]; //NOTE: unoptomized memory access.
    r.dir.y = ray_buffer[ray_offset+1];
    r.dir.z = ray_buffer[ray_offset+2];



    union {
		float f;
		unsigned int ui;
	} res;

    res.f = (float)magic*M_PI+x;//get some decimals.
    unsigned int seed1 = res.ui + (int)(sin((float)x)*7.f);

    res.f = (float)magic*M_PI+y;//get some decimals.
    unsigned int seed2 = res.ui + (int)(sin((float)y)*7.f);

    //unsigned int seed1 = (int)((magic+1)%x)*7+magic+x/*sin((float)magic))*/,
    //    seed2 = (int)((magic+1)%y)*7+magic+y/*cos((float)magic))*/;
    vec3 fin_colour = (vec3)(0.0f, 0.0f, 0.0f);
    for(int i = 0; i < NUM_SAMPLES; i++)
    {
        vec3 accum_color = (vec3)(0.0f, 0.0f, 0.0f);
        vec3 mask        = (vec3)(1.0f, 1.0f, 1.0f);
        ray sr = r;

        for(int bounces = 0; bounces < NUM_BOUNCES; bounces++)
        {
            collision_result result;
            if(!collide_all(sr, &result, material_buffer, spheres, planes, meshes, indices,
                            vertices, normals))
            {
                accum_color += mask * sky.xyz;//(vec3) (1.f,1.f,1.f);//(vec3)(0.15f, 0.15f, 0.25f);
                break;
            }


            float rand1 = get_random(&seed1, &seed2);
            float rand2 = get_random(&seed1, &seed2); //seed but even more seedy :)
            // float3 normal_facing = dot(result.normal, sr.dir) < 0.0f ? result.normal : result.normal * (-1.0f);


            vec3 sample_dir =  cosineSampleHemisphere(rand1, rand2, result.normal);

            sr.orig = result.point + result.normal * 0.0001f; //sweet spot for epsilon
            sr.dir = sample_dir;

            accum_color += mask * (float)(result.mat.reflectivity==1.)*2; //NOTE: EMMISION


            mask *= result.mat.colour;

            mask *= dot(sample_dir, result.normal);

            //vec4 colour = shade(result);

        }

        fin_colour += clamp(accum_color,0.f,1.f) * (1.f/NUM_SAMPLES);
    }

    //   colour = mix(colour, colours[0],  result.mat.reflectivity);

    out_tex[x+y*width] = (vec4) (fin_colour, 0);//get_colour((vec4) (fin_colour, 0) );
    //THIS EMPTY LINE IS NECESSARY AND PREVENTS IT FROM CRASHING (not joking) PLS HELP NVIDIA.

}


__kernel void buffer_average(
    __global uchar4* out_tex,
    __global uchar4* fresh_frame_tex,
    const unsigned int width,
    const unsigned int height,
    const unsigned int sample
    /*const unsigned int num_samples*/)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width);

    //t_tex[offset] = (char4) (mi( (float4)fresh_frame_tex[offset], (float4)out_tex[offset], (float)sample/255))

    float4 temp = mix((float4)(
                          (float)fresh_frame_tex[offset].x,
                          (float)fresh_frame_tex[offset].y,
                          (float)fresh_frame_tex[offset].z,
                          (float)fresh_frame_tex[offset].w),
                      (float4)(
                          (float)out_tex[offset].x,
                          (float)out_tex[offset].y,
                          (float)out_tex[offset].z,
                          (float)out_tex[offset].w), (float)sample/24.f);
    /*vec4 temp =  (float)(
        (float)fresh_frame_tex[offset].x,
        (float)fresh_frame_tex[offset].y,
        (float)fresh_frame_tex[offset].z,
        (float)fresh_frame_tex[offset].w)/12.f;*/
    out_tex[offset] = (uchar4) ((unsigned char)temp.x,
                                (unsigned char)temp.y,
                                (unsigned char)temp.z,
                                (unsigned char)temp.w);
/*
        fresh_frame_tex[offset]/(unsigned char)(1.f/(1-(float)sample/255))
        + out_tex[offset]/(unsigned char)(1.f/((float)sample/255));*/
}

__kernel void f_buffer_average(
    __global vec4* out_tex,
    __global vec4* fresh_frame_tex,
    const unsigned int width,
    const unsigned int height,
    const unsigned int num_samples,
    const unsigned int sample)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width);
    out_tex[offset] = mix(fresh_frame_tex[offset], out_tex[offset], ((float)sample+1.f)/(float)num_samples);
}

__kernel void f_buffer_to_byte_buffer(
    __global unsigned int* out_tex,
    __global vec4* fresh_frame_tex,
    const unsigned int width,
    const unsigned int height)
{
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = (x + y * width);
    //if(fresh_frame_tex[offset].x!=fresh_frame_tex[offset].y)printf("n (%f,%f,%f)\n", fresh_frame_tex[offset].x,fresh_frame_tex[offset].y,fresh_frame_tex[offset].z);
    out_tex[offset] = get_colour(fresh_frame_tex[offset]);
}
