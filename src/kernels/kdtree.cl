#define KDTREE_LEAF 1
#define KDTREE_NODE 2

typedef struct
{
    uchar type;
    uint num_triangles;
} kd_tree_leaf_template;

typedef struct
{
    uchar type;

    uint num_triangles;
    ulong triangle_start;
} kd_tree_leaf; //TODO: align

typedef struct
{
    uchar type;
    uchar k;
    float b;

    ulong left_index;
    ulong right_index;
} kd_tree_node; //TODO: align


typedef union a_vec3
{
    vec3 v;
    float a[4];
} a_vec3;

//TODO; put this type in another file
/*typedef struct kd_ray
{
    union //fuck
    {
        vec4  origin;
        float origin_a[4]; //works alright for now, but there are faster ways of doing it.
    };
    union
    {
        vec4  direction;
        float direction_a[4];
    };
} kd_ray;*/

typedef struct kd_stack_elem
{
    ulong node;
    float min;
    float max;
} kd_stack_elem;

typedef vec4 kd_44_matrix[4];


/*float get_component(vec4 v, unsigned char k)
{
    const unsigned int const_masks[4][4] =
        {
            {0xffffffff, 0, 0, 0},
            {0, 0xffffffff, 0, 0},
            {0, 0, 0xffffffff, 0},
            {0, 0, 0, 0xffffffff}
        };
    vec4 nv = v & const_masks[k];
    return nv.x + nv.y + nv.z + nv.z;
}*/


void kd_update_state(const __global char* kd_tree, ulong indx, uchar* type, kd_tree_node* node, kd_tree_leaf* leaf)
{
    *type = *(kd_tree+indx);
    if(*type == KDTREE_LEAF)
    {
        *leaf = *((__global kd_tree_leaf*) (kd_tree + indx));
        leaf->triangle_start = indx + sizeof(kd_tree_leaf_template);
    }
    else
        *node = *((__global kd_tree_node*) (kd_tree + indx));
}

inline float get_elem(vec3 v, unsigned char k, __global kd_44_matrix* mask)
{
    vec3 nv = v * (*mask)[k].xyz;
    return nv.x + nv.y + nv.z;
}

//#define B 3*32 //batch size
#define STACK_SIZE 32 //tune later
#define LOAD_BALANCER_BATCH_SIZE        3*32

//USE: CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
__kernel void kdtree_intersection(
    __global float* out_buf,
    const unsigned int buf_offset, //offset and stride allows us to write into the ray_buffer with mesh indices
    const unsigned int buf_stride,
    const __global ray* ray_buffer, //TODO: make vec4

    __global uint* dumb_data, //NOTE: REALLY DUMB, you can't just have a global variable in ocl.


//Mesh
    const __global mesh* meshes,
    image1d_buffer_t     indices,
    image1d_buffer_t     vertices,
    image1d_buffer_t     normals,
    const __global char* kd_tree,

    const unsigned int num_rays,
    const vec4 pos)
{


    const uint blocksize_x = get_local_size(0); //should be 32
    const uint blocksize_y = get_local_size(1);

    uint x = get_local_id(0);
    uint y = get_local_id(1);

    __local volatile int next_ray_array[BLOCKSIZE_Y]; //TODO: make fix sized
    __local volatile int ray_count_array[BLOCKSIZE_Y]; //TODO: make a macro
    next_ray_array[get_local_id(1)]  = 0;
    ray_count_array[get_local_id(1)] = 0;


    kd_stack_elem stack[STACK_SIZE]; //honestly im not sure about the plus three, it seems unnecessary (micro optomization)
    uint stack_length = 0;

    __global uint* warp_counter = dumb_data;
    __global kd_44_matrix* elem_mask = (__global kd_44_matrix*)(dumb_data+1);
    //traversal_stack[STACK_SIZE+0] = get_local_id(0); //saves a register I guess, was recommended in src.
    //traversal_stack[STACK_SIZE+1] = get_local_id(1);


    ray r;
    float t_hit = 1000000; //INFINITY
    float t_min, t_max;
    float scene_t_min, scene_t_max;
    kd_tree_node node;
    kd_tree_leaf leaf;
    uchar current_type = KDTREE_NODE;
    bool pushdown = false;
    kd_tree_node root;

    //t_min = t_max = min());//(vec3) SCENE_MIN; //SCENE_MIN is a macro

    while(true)
    {
        uint tidx = x;//stack[STACK_SIZE + 0];
        uint widx = y;//stack[STACK_SIZE + 1];
        __local volatile int* local_pool_ray_count = ray_count_array+widx;
        __local volatile int* local_pool_next_ray  = next_ray_array+widx;

        //Grab new rays
        if(tidx == 0 && *local_pool_ray_count <= 0) //only the first work item gets memory
        {
            *local_pool_next_ray = atomic_add(warp_counter, LOAD_BALANCER_BATCH_SIZE);
            *local_pool_ray_count = LOAD_BALANCER_BATCH_SIZE;
        }

        {
            uint ray_indx = *local_pool_next_ray + tidx;
            if(ray_indx >= num_rays) //ray index is past num rays, work is done
                break;

            if(tidx == 0)
            {
                *local_pool_next_ray += 32;
                *local_pool_ray_count -= 32;
            }

            r = ray_buffer[ray_indx];

            //flaot scene_min_t = min(((vec3)SCENE_MIN - ray.origin) * );



            t_hit = 100000000; //infinity
            if(!getTBoundingBox(SCENE_MIN, SCENE_MAX, r, &scene_t_min, &scene_t_max)) //SCENE_MIN is a macro
                printf("Shit\n");
            t_max = t_min = scene_t_min;

            root = *((__global kd_tree_node*) kd_tree);
        }

        while(t_max < scene_t_max)
        {
            if(stack_length == 0)
            {
                node  = root; //root
                t_min = t_max;
                t_max = scene_t_max; //macro
                pushdown = true;
            }
            else
            { //pop

                //TODO: update this
                //WRONG: node  = stack[stack_length-1].node;
                t_min = stack[stack_length-1].min;
                t_max = stack[stack_length-1].max;
                stack_length--;
                pushdown = false;
            }
            while(current_type != KDTREE_LEAF)
            {
                unsigned char k = node.k;
                float t_split = (node.b - get_elem(r.orig, k, elem_mask)) / get_elem(r.dir, k, elem_mask);
                ulong first  = (get_elem(r.dir, k, elem_mask)>0 ?
                                (node.left_index) :  (node.right_index));
                ulong second = (get_elem(r.dir, k, elem_mask)>0 ?
                                (node.right_index) :  (node.left_index));

                if( t_split >= t_max || t_split < 0)
                    kd_update_state(kd_tree, first, &current_type, &node, &leaf);
                else if(t_split <= t_min)
                    kd_update_state(kd_tree, second, &current_type, &node, &leaf);
                else
                {
                    //update
                    stack[stack_length++] = (kd_stack_elem) {second, t_split, t_max}; //push
                    kd_update_state(kd_tree, first, &current_type, &node, &leaf);

                    t_max = t_split;
                    pushdown = false;
                }
                if(pushdown)
                    root = node;//UPDATE
            }
            //Found leaf
            for(ulong t = leaf.triangle_start; t < (ulong)leaf.num_triangles+leaf.triangle_start; t++)
            {
                vec3 tri[4];
                unsigned int index_offset = *(__global uint*)(kd_tree+t);
                //get vertex (first element of each index)
                int4 idx_0 = read_imagei(indices, index_offset+0);
                int4 idx_1 = read_imagei(indices, index_offset+1);
                int4 idx_2 = read_imagei(indices, index_offset+2);

                tri[0] = read_imagef(vertices, idx_0.x).xyz;
                tri[1] = read_imagef(vertices, idx_1.x).xyz;
                tri[2] = read_imagef(vertices, idx_2.x).xyz;

                vec3 hit_coords; // t u v
                if(does_collide_triangle(tri, &hit_coords, r))
                {
                    t_hit = min(t_hit, hit_coords.x); //t
                    if(t_hit < t_min) // goes by closest to furthest, so if it hits it will be the closest
                    {//early exit
                        break; //TODO: do something
                    }
                }
            }

        }

    }

}
