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
