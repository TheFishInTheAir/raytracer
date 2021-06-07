#define KDTREE_LEAF 1
#define KDTREE_NODE 2

typedef struct
{
    uchar type;

    uint num_triangles;
} __attribute__ ((aligned (16))) kd_tree_leaf_template;

typedef struct
{
    uchar type;

    uint num_triangles;
    ulong triangle_start;
} kd_tree_leaf;

typedef struct
{
    uchar type;
    uchar k;
    float b;

    ulong left_index;
    ulong right_index;
} __attribute__ ((aligned (16))) kd_tree_node;

void dbg_print_node(kd_tree_node n)
{
    printf("\nNODE: type: %u, k: %u, b: %f, l: %llu, r: %llu \n",
           (unsigned int) n.type, (unsigned int) n.k, n.b,
           n.left_index, n.right_index);
}

typedef union a_vec3
{
    vec3 v;
    float a[4];
} a_vec3;

typedef struct kd_stack_elem
{
    ulong node;
    float min;
    float max;
} kd_stack_elem;

void kd_update_state(__global long* kd_tree, ulong indx, uchar* type,
                     kd_tree_node* node, kd_tree_leaf* leaf)
{
    *type = *((__global uchar*)(kd_tree+indx));

    if(*type == KDTREE_LEAF)
    {
        kd_tree_leaf_template template = *((__global kd_tree_leaf_template*) (kd_tree + indx));
        leaf->type = template.type;
        leaf->num_triangles = template.num_triangles;

        leaf->triangle_start = indx + sizeof(kd_tree_leaf_template)/8;
    }
    else
        *node = *((__global kd_tree_node*) (kd_tree + indx));

}

inline float get_elem(vec3 v, uchar k, kd_44_matrix mask)
{
    k = min(k, (uchar)2);

    //NOTE: it has to be MSB on the mask
    vec3 nv = select((vec3)(0), v, mask[k].xyz);

    return nv.x + nv.y + nv.z;
}

//Tune later
#define STACK_SIZE 16
#define LOAD_BALANCER_BATCH_SIZE        32*3

//NOTE: Ideas for improvement
// Not compeltely sure how effective barriers are in this current setup,
// might make sense to have add a branch in main persistent threading work loop
// that doesnt do any work but would allow the GPU Work Group to always hit every memory barrier
// and only stop the work loop once the entire work group is finished.
// This could potentially improve parallelism within the work group.



__kernel void kdtree_intersection(
    __global kd_tree_collision_result* out_buf,
    __global ray* ray_buffer,

    //NOTE: super annoying, can't JUST have a global variable in ocl.
    __global uint* dumb_data,

    //Mesh
    __global mesh* meshes,
    image1d_buffer_t     indices,
    image1d_buffer_t     vertices,

    //TODO: use a higher allignment type
    __global long* kd_tree,

    unsigned int num_rays)
{

    const uint blocksize_x = BLOCKSIZE_X;
    const uint blocksize_y = BLOCKSIZE_Y;

    //NOTE: not technically correct, but kinda is
    uint x = get_local_id(0) % BLOCKSIZE_X; //id within the warp
    uint y = get_local_id(0) / BLOCKSIZE_X; //id of the warp in the SM

    __local volatile int next_ray_array[BLOCKSIZE_Y];
    __local volatile int ray_count_array[BLOCKSIZE_Y];
    next_ray_array[y]  = 0;
    ray_count_array[y] = 0;

    kd_stack_elem stack[STACK_SIZE];
    uint stack_length = 0;

    //NOTE: Reminder to make sure vectors always have to be alligned
    kd_44_matrix elem_mask = (kd_44_matrix)(dumb_data);
    __global uint* warp_counter = dumb_data+16;


    //NOTE: this block of variables is probably pretty bad for the cache
    ray           r;
    float         t_hit = INFINITY;
    vec2          hit_info = (vec2)(0,0);
    unsigned int  tri_indx;
    float         t_min, t_max;
    float         scene_t_min = 0, scene_t_max = INFINITY;
    kd_tree_node  node;
    kd_tree_leaf  leaf;
    uchar         current_type = KDTREE_NODE;
    bool          pushdown = false;
    kd_tree_node  root;
    uint          ray_indx;

    while(true)
    {
        uint tidx = x; // SINGLE WARPS WORTH OF WORK 0-32
        uint widx = y; // WARPS PER SM 0-4 (for example)

        //get warp ids pool
        __local volatile int* local_pool_ray_count = ray_count_array+widx;
        __local volatile int* local_pool_next_ray  = next_ray_array+widx;

        scene_t_min = 0;
        scene_t_max = INFINITY;
        hit_info = (vec2)(0,0);
        bool did_early_exit = false;
        current_type = KDTREE_NODE;
        pushdown = false;

        //Grab new rays
        //only the first work (of the pool) item gets memory
        if(tidx == 0 && *local_pool_ray_count <= 0)
        {
            //batch complete
            *local_pool_next_ray = atomic_add(warp_counter, LOAD_BALANCER_BATCH_SIZE);

            *local_pool_ray_count = LOAD_BALANCER_BATCH_SIZE;
        }

        //Try to synchronize warp on each iteration....
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

        {

            ray_indx = *local_pool_next_ray + tidx;
            barrier(CLK_LOCAL_MEM_FENCE);

            //ray index is past num rays, work is done
            if(ray_indx >= num_rays)
                break;

            //NOTE: this doesn't guarentee
            if(tidx == 0)
            {
                *local_pool_next_ray  += BLOCKSIZE_X;
                *local_pool_ray_count -= BLOCKSIZE_X;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            r = ray_buffer[ray_indx];

            t_hit = INFINITY;

            if(!getTBoundingBox((vec3) SCENE_MIN, (vec3) SCENE_MAX, r, &scene_t_min, &scene_t_max))
            {
                scene_t_max = -INFINITY;
            }


            t_max = t_min = scene_t_min;

            stack_length = 0;
            root = *((__global kd_tree_node*) kd_tree);
        }
        stack_length = 0;

        while(t_max < scene_t_max && !did_early_exit)
        {

            if(stack_length == (uint) 0)
            {
                //Bring to root
                node  = root;
                current_type = KDTREE_NODE;
                t_min = t_max;
                t_max = scene_t_max;
                pushdown = true;
            }
            else
            { //pop

                t_min = stack[stack_length-1].min;
                t_max = stack[stack_length-1].max;
                kd_update_state(kd_tree, stack[stack_length-1].node, &current_type, &node, &leaf);

                stack_length--;
                pushdown = false;
            }


            while(current_type != KDTREE_LEAF)
            {
                unsigned char k = node.k;

                float t_split = (node.b - get_elem(r.orig, k, elem_mask)) /
                                 get_elem(r.dir, k, elem_mask);

                bool left_close =
                    (get_elem(r.orig, k, elem_mask) < node.b) ||
                    (get_elem(r.orig, k, elem_mask) == node.b && get_elem(r.dir, k, elem_mask) <= 0);
                ulong thing = left_close ? 0xffffffffffffffff : 0;
                ulong first  = select(node.right_index, node.left_index,
                                      thing);
                ulong second = select(node.left_index, node.right_index,
                                      thing);


                kd_update_state(kd_tree,
                                ( t_split > t_max || t_split <= 0)||!(t_split < t_min) ? first : second,
                                &current_type, &node, &leaf);

                if( !(t_split > t_max || t_split <= 0) && !(t_split < t_min))
                {
                    stack[stack_length++] = (kd_stack_elem) {second, t_split, t_max}; //push
                    t_max = t_split;
                    pushdown = false;
                }


                root = pushdown ? node : root;

            }

            //Found leaf
            for(ulong t = 0; t <leaf.num_triangles; t++)
            {
                vec3 tri[4];
                unsigned int index_offset =
                    *((__global uint*)(kd_tree+leaf.triangle_start)+t);

                //get vertex (first element of each index)
                //NOTE: issue with unsigned ints on osx, works on all other platforms though smh
                int4 idx_0 = read_imagei(indices, (int)index_offset+0);
                int4 idx_1 = read_imagei(indices, (int)index_offset+1);
                int4 idx_2 = read_imagei(indices, (int)index_offset+2);

                tri[0] = read_imagef(vertices, (int)idx_0.x).xyz;
                tri[1] = read_imagef(vertices, (int)idx_1.x).xyz;
                tri[2] = read_imagef(vertices, (int)idx_2.x).xyz;

                vec3 hit_coords; // t u v
                if(does_collide_triangle(tri, &hit_coords, r))
                {
                    if(hit_coords.x<=0)
                        continue;
                    if(hit_coords.x < t_hit)
                    {
                        t_hit = hit_coords.x;     //t
                        hit_info = hit_coords.yz; //u v
                        tri_indx = index_offset;

                        // goes by closest to furthest, so if it hits it will be the closest
                        if(t_hit < t_max)
                        {//early exit

                            did_early_exit = true;
                            break;
                        }

                    }

                }

            }

        }
        did_early_exit = false;

        //By this point a triangle will have been found.
        kd_tree_collision_result result = {0};

        if(!isinf(t_hit))//if t_hit != INFINITY
        {
            result.triangle_index = tri_indx;
            result.t = t_hit;
            result.u = hit_info.x;
            result.v = hit_info.y;
        }

        out_buf[ray_indx] = result;
    }

}

__kernel void kdtree_ray_draw(
    __global unsigned int* out_tex,
    __global ray* rays,

    const unsigned int width)
{
    const vec4 sky = (vec4) (0.84, 0.87, 0.93, 0);
    //return;
    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;

    ray r = rays[offset];

    r.orig = (r.orig+1) / 2;

    out_tex[offset] = get_colour( (vec4) (r.orig,1) );
}


__kernel void kdtree_test_draw(
    __global unsigned int* out_tex,
    __global kd_tree_collision_result* kd_results,

    const __global material* material_buffer,
    //meshes
    __global mesh* meshes,

    image1d_buffer_t     indices,
    image1d_buffer_t     vertices,
    image1d_buffer_t     normals,
    const unsigned int width)
{
    const vec4 sky = (vec4) (0.84, 0.87, 0.93, 0);

    int id = get_global_id(0);
    int x  = id%width;
    int y  = id/width;
    int offset = x+y*width;

    kd_tree_collision_result res = kd_results[offset];
    if(res.t==0)
    {
        out_tex[offset] = get_colour( (vec4) (0) );
        return;
    }
    int4 i1 = read_imagei(indices, (int)res.triangle_index);
    int4 i2 = read_imagei(indices, (int)res.triangle_index+1);
    int4 i3 = read_imagei(indices, (int)res.triangle_index+2);
    mesh m = meshes[i1.w];
    material mat = material_buffer[m.material_index];

    vec3 normal =
        read_imagef(normals, (int)i1.y).xyz*(1-res.u-res.v)+
        read_imagef(normals, (int)i2.y).xyz*res.u+
        read_imagef(normals, (int)i3.y).xyz*res.v;

    normal = (normal+1) / 2;

    out_tex[offset] = get_colour( (vec4) (normal,1) );
}
