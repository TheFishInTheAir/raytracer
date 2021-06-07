/*********/
/* Types */
/*********/

#define MESH_SCENE_DATA_PARAM image1d_buffer_t indices, image1d_buffer_t vertices, image1d_buffer_t normals
#define MESH_SCENE_DATA       indices, vertices, normals

typedef struct //16 bytes
{
    vec3 colour;

    float reflectivity;
} __attribute__ ((aligned (16))) material;

typedef struct
{
    vec3 orig;
    vec3 dir;
} ray;

typedef struct
{
    bool did_hit;
    vec3 normal;
    vec3 point;
    float dist;
    material mat;
} collision_result;

typedef struct //32 bytes (one word)
{
    vec3 pos;
    //4 bytes padding
    float radius;
    int material_index;
    //8 bytes padding
} __attribute__ ((aligned (16))) sphere;

typedef struct plane
{
    vec3 pos;
    vec3 normal;

    int material_index;
} __attribute__ ((aligned (16))) plane;

typedef struct
{

    mat4 model;

    vec3 max;
    vec3 min;

    int index_offset;
    int num_indices;


    int material_index;
} __attribute__((aligned (32))) mesh; //TODO: align with cpu NOTE: I don't think we need 32

typedef struct
{
    const __global material* material_buffer;
    const __global sphere* spheres;
    const __global plane* planes;
    //Mesh
    const __global mesh* meshes;
} scene;

bool getTBoundingBox(vec3 vmin, vec3 vmax,
                     ray r, float* tmin, float* tmax) //NOTE: could be wrong
{

    vec3 invD = 1/r.dir;///vec3(1/dir.x, 1/dir.y, 1/dir.z);
	vec3 t0s = (vmin - r.orig) * invD;
  	vec3 t1s = (vmax - r.orig) * invD;

  	vec3 tsmaller = min(t0s, t1s);
    vec3 tbigger  = max(t0s, t1s);

    *tmin = max(*tmin, max(tsmaller.x, max(tsmaller.y, tsmaller.z)));
    *tmax = min(*tmax, min(tbigger.x,  min(tbigger.y, tbigger.z)));

	return (*tmin < *tmax);

    /* vec3 tmin = (vmin - r.orig) / r.dir; */
    /* vec3 tmax = (vmax - r.orig) / r.dir; */

    /* vec3 real_min = min(tmin, tmax); */
    /* vec3 real_max = max(tmin, tmax); */

    /* vec3 minmax = min(min(real_max.x, real_max.y), real_max.z); */
    /* vec3 maxmin = max(max(real_min.x, real_min.y), real_min.z); */

    /* if (dot(minmax,minmax) >= dot(maxmin, maxmin)) */
    /* { */
    /*     *t_min = sqrt(dot(maxmin,maxmin)); */
    /*     *t_max = sqrt(dot(minmax,minmax)); */
    /*     return (dot(maxmin, maxmin) > 0.001f ? true : false); */
    /* } */
    /* else return false; */
}


bool hitBoundingBox(vec3 vmin, vec3 vmax,
                    ray r)
{
    vec3 tmin = (vmin - r.orig) / r.dir;
    vec3 tmax = (vmax - r.orig) / r.dir;

    vec3 real_min = min(tmin, tmax);
    vec3 real_max = max(tmin, tmax);

    vec3 minmax = min(min(real_max.x, real_max.y), real_max.z);
    vec3 maxmin = max(max(real_min.x, real_min.y), real_min.z);

    if (dot(minmax,minmax) >= dot(maxmin, maxmin))
    { return (dot(maxmin, maxmin) > 0.001f ? true : false); }
    else return false;
}



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

//tri has extra for padding
bool does_collide_triangle(vec3 tri[4], vec3* hit_coords, ray r)
{

    vec3 ab = tri[1] - tri[0];
    vec3 ac = tri[2] - tri[0];

    //Triple product
    vec3 pvec = cross(r.dir, ac);
    float det = dot(ab, pvec);

    // Behind or close to parallel.
    if (det < EPSILON)
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

bool does_collide_sphere(sphere s, ray r, float *dist)
{
    // solutions for t if the ray intersects
    float t0, t1;

    // analytic solution
    vec3 L = s.pos- r.orig;
    float b = dot(r.dir, L) ;//* 2.0f;
    //NOTE: you can optimize out the square.
    float c = dot(L, L) - (s.radius*s.radius);

    // discriminant of quadratic formula
    float disc = b * b - c/**a*/;

    // solve for t (distance to hitpoint along ray)
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

bool does_collide_plane(plane p, ray r, float *dist)
{
    float denom = dot(r.dir, p.normal);
    //Counter intuitive.
    if (denom < EPSILON)
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


bool does_collide_with_mesh(mesh collider, ray r, vec3* normal, float* dist, scene s,
                            MESH_SCENE_DATA_PARAM)
{
    *dist = FAR_PLANE;
    float min_t = FAR_PLANE;
    vec3 hit_coord; //NOTE: currently unused
    ray r2 = r;
    if(!hitBoundingBox(collider.min, collider.max, r))
    {
        return false;
    }

    // each ivec3
    for(int i = 0; i < collider.num_indices/3; i++)
    {
        vec3 tri[4];

        //get vertex (first element of each index)

        int4 idx_0 = read_imagei(indices, i*3+collider.index_offset+0);
        int4 idx_1 = read_imagei(indices, i*3+collider.index_offset+1);
        int4 idx_2 = read_imagei(indices, i*3+collider.index_offset+2);

        tri[0] = read_imagef(vertices, idx_0.x).xyz;
        tri[1] = read_imagef(vertices, idx_1.x).xyz;
        tri[2] = read_imagef(vertices, idx_2.x).xyz;

        vec3 bc_hit_coords = (vec3)(0.f); //t u v = x y z
        if(does_collide_triangle(tri, &bc_hit_coords, r) &&
           bc_hit_coords.x<min_t && bc_hit_coords.x>0)
        {
            min_t = bc_hit_coords.x; //t (distance along direction)
            *normal =
                read_imagef(normals, idx_0.y).xyz*(1-bc_hit_coords.y-bc_hit_coords.z)+
                read_imagef(normals, idx_1.y).xyz*bc_hit_coords.y+
                read_imagef(normals, idx_2.y).xyz*bc_hit_coords.z;
            //break; //convex optimization
        }

    }


    *dist = min_t;
    return min_t != FAR_PLANE;

}

bool does_collide_with_mesh_nieve(mesh collider, ray r, vec3* normal, float* dist, scene s,
                                  image1d_buffer_t tree, MESH_SCENE_DATA_PARAM)
{
    *dist = FAR_PLANE;
    float min_t = FAR_PLANE;
    vec3 hit_coord; //NOTE: currently unused
    ray r2 = r;
    if(!hitBoundingBox(collider.min, collider.max, r))
    {
        return false;
    }

    // each ivec3
    for(int i = 0; i < collider.num_indices/3; i++)
    {
        vec3 tri[4];

        //get vertex (first element of each index)

        int4 idx_0 = read_imagei(indices, i*3+collider.index_offset+0);
        int4 idx_1 = read_imagei(indices, i*3+collider.index_offset+1);
        int4 idx_2 = read_imagei(indices, i*3+collider.index_offset+2);

        tri[0] = read_imagef(vertices, idx_0.x).xyz;
        tri[1] = read_imagef(vertices, idx_1.x).xyz;
        tri[2] = read_imagef(vertices, idx_2.x).xyz;

        vec3 bc_hit_coords = (vec3)(0.f); //t u v = x y z
        if(does_collide_triangle(tri, &bc_hit_coords, r) &&
           bc_hit_coords.x<min_t && bc_hit_coords.x>0)
        {
            min_t = bc_hit_coords.x; //t (distance along direction)
            *normal =
                read_imagef(normals, idx_0.y).xyz*(1-bc_hit_coords.y-bc_hit_coords.z)+
                read_imagef(normals, idx_1.y).xyz*bc_hit_coords.y+
                read_imagef(normals, idx_2.y).xyz*bc_hit_coords.z;
                //break; //convex optimization
        }

    }


    *dist = min_t;
    return min_t != FAR_PLANE;
}

bool does_collide_with_mesh_alt(mesh collider, ray r, vec3* normal, float* dist, scene s,
                            MESH_SCENE_DATA_PARAM)
{
    *dist = FAR_PLANE;
    float min_t = FAR_PLANE;
    vec3 hit_coord; //NOTE: currently unused
    ray r2 = r;

    for(int i = 0; i < SCENE_NUM_INDICES/3; i++)
    {
        vec3 tri[4];

        //get vertex (first element of each index)

        int4 idx_0 = read_imagei(indices, i*3+collider.index_offset+0);
        int4 idx_1 = read_imagei(indices, i*3+collider.index_offset+1);
        int4 idx_2 = read_imagei(indices, i*3+collider.index_offset+2);

        tri[0] = read_imagef(vertices, idx_0.x).xyz;
        tri[1] = read_imagef(vertices, idx_1.x).xyz;
        tri[2] = read_imagef(vertices, idx_2.x).xyz;


        vec3 bc_hit_coords = (vec3)(0.f); //t u v = x y z
        if(does_collide_triangle(tri, &bc_hit_coords, r) &&
           bc_hit_coords.x<min_t && bc_hit_coords.x>0)
        {
                min_t = bc_hit_coords.x; //t (distance along direction)
                *normal =
                    read_imagef(normals, idx_0.y).xyz*(1-bc_hit_coords.y-bc_hit_coords.z)+
                    read_imagef(normals, idx_1.y).xyz*bc_hit_coords.y+
                    read_imagef(normals, idx_2.y).xyz*bc_hit_coords.z;
        }

    }


    *dist = min_t;
    return min_t != FAR_PLANE;

}



/************************/
/* High Level Collision */
/************************/


bool collide_meshes(ray r, collision_result* result, scene s, MESH_SCENE_DATA_PARAM)
{

    float dist = FAR_PLANE;
    result->did_hit = false;
    result->dist = FAR_PLANE;

    for(int i = 0; i < SCENE_NUM_MESHES; i++)
    {
        mesh current_mesh = s.meshes[i];
        float local_dist = FAR_PLANE;
        vec3 normal;
        if(does_collide_with_mesh(current_mesh, r, &normal,  &local_dist, s, MESH_SCENE_DATA))
        {

            if(local_dist<dist)
            {
                dist = local_dist;
                result->dist = dist;
                result->normal = normal;
                result->point = (r.dir*dist)+r.orig;
                result->mat = s.material_buffer[current_mesh.material_index];
                result->did_hit = true;
            }
        }
    }
    return result->did_hit;
}

bool collide_primitives(ray r, collision_result* result, scene s)
{

    float dist = FAR_PLANE;
    result->did_hit = false;
    result->dist = FAR_PLANE;
    for(int i = 0; i < SCENE_NUM_SPHERES; i++)
    {
        sphere current_sphere = s.spheres[i];
        float local_dist = FAR_PLANE;
        if(does_collide_sphere(current_sphere, r, &local_dist))
        {
            if(local_dist<dist)
            {
                dist = local_dist;
                result->did_hit = true;
                result->dist    = dist;
                result->point   = r.dir*dist+r.orig;
                result->normal  = normalize(result->point - current_sphere.pos);
                result->mat     = s.material_buffer[current_sphere.material_index];
            }
        }
    }

    for(int i = 0; i < SCENE_NUM_PLANES; i++)
    {
        plane current_plane = s.planes[i];
        float local_dist =  FAR_PLANE;
        if(does_collide_plane(current_plane, r, &local_dist))
        {
            if(local_dist<dist)
            {
                dist = local_dist;
                result->did_hit = true;
                result->dist    = dist;
                result->point   = r.dir*dist+r.orig;
                result->normal  = current_plane.normal;
                result->mat     = s.material_buffer[current_plane.material_index];
            }
        }
    }

    return dist != FAR_PLANE;
}

bool collide_all(ray r, collision_result* result, scene s, MESH_SCENE_DATA_PARAM)
{
    float dist = FAR_PLANE;
    if(collide_primitives(r, result, s))
        dist = result->dist;

    collision_result m_result;
    if(collide_meshes(r, &m_result, s, MESH_SCENE_DATA))
        if(m_result.dist < dist)
            *result = m_result;

    return result->did_hit;
}
