#define KDTREE_LEAF 1
#define KDTREE_NODE 2

//TODO: put in util
#define DEBUG
#ifdef DEBUG
//NOTE: this will be slow. NOTE: the barrier just hangs the program so we can see the error
#define assert(x)                                                       \
    if (! (x))                                                          \
    {                                                                   \
        int i = 0;while(i++ < 100)printf("Assert(%s) failed in %s:%d\n", #x, __FUNCTION__, __LINE__); \
        return;                                                        \
    }
#else
#define assert(x)  //NOTHING
#endif
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

typedef __global uint4* kd_44_matrix;


void kd_update_state(__global char* kd_tree, ulong indx, uchar* type,
                     kd_tree_node* node, kd_tree_leaf* leaf)
{


    *type = *((__global uchar*)(kd_tree+indx));
    //assert(*type<=2);

    //assert(indx%8 == 0);

    if(*type == KDTREE_LEAF)
    {
        kd_tree_leaf_template template = *((__global kd_tree_leaf_template*) (kd_tree + indx));
        leaf->type = template.type;
        leaf->num_triangles = template.num_triangles;
        //if(template.num_triangles>0)
        //    printf("%d %u test\n", (int) template.type, template.num_triangles);
        leaf->triangle_start = indx + sizeof(kd_tree_leaf_template);
    }
    else
        *node = *((__global kd_tree_node*) (kd_tree + indx));

}

void dbg_print_node(kd_tree_node n)
{
    printf("\nNODE: type: %u, k: %u, b: %f, l: %llu, r: %llu \n",
           (unsigned int) n.type, (unsigned int) n.k, n.b,
           n.left_index, n.right_index);
}

void dbg_print_matrix(kd_44_matrix m)
{
    //printf("IS IT GONNA WORK?\n");
    //printf("IS IT GONNA WORK pt 2? ---> %f \n", m[0].x);
    //printf("%v3u\n", m[0]);

    /*printf("[%u %u %u %u]\n",
           m[0].x, m[0].y, m[0].z, m[0].w);
    */


    printf("[%2u %2u %2u %2u]\n" \
           "[%2u %2u %2u %2u]\n" \
           "[%2u %2u %2u %2u]\n" \
           "[%2u %2u %2u %2u]\n\n",
           m[0].x, m[0].y, m[0].z, m[0].w,
           m[1].x, m[1].y, m[1].z, m[1].w,
           m[2].x, m[2].y, m[2].z, m[2].w,
           m[3].x, m[3].y, m[3].z, m[3].w);
}

inline float get_elem(vec3 v, uchar k, kd_44_matrix mask)
{
    k = min(k,(uchar)2);
    vec3 nv = select((vec3)(0), v, mask[k].xyz);//NOTE: it has to be MSB on the mask
    //if(v.x!=0.f)
    //    printf("test: %f %f %f [%d]: (%u %u %u %u) %f\n", v.x, v.y, v.z, (int) k, mask[k].x, mask[k].y, mask[k].z, mask[k].w, nv.x + nv.y + nv.z);
    return nv.x + nv.y + nv.z;
}

//#define B 3*32 //batch size
#define STACK_SIZE 16 //tune later
#define LOAD_BALANCER_BATCH_SIZE        32

//USE: CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE TODO: add const
__kernel void kdtree_intersection(
    __global kd_tree_collision_result* out_buf,
    __global ray* ray_buffer, //TODO: make vec4

    __global uint* dumb_data, //NOTE: REALLY DUMB, you can't just have a global variable in ocl.

//Mesh
    __global mesh* meshes,
    image1d_buffer_t     indices,
    image1d_buffer_t     vertices,
    __global char* kd_tree,   //TODO: higher allignment type

    unsigned int num_rays)
{

    const uint blocksize_x = BLOCKSIZE_X; //should be 32 //NOTE: REMOVED A THING
    const uint blocksize_y = BLOCKSIZE_Y;

    uint x = get_local_id(0) % BLOCKSIZE_X; //id within the warp
    uint y = get_local_id(0) / BLOCKSIZE_X; //id of the warp in the SM

    __local volatile int next_ray_array[BLOCKSIZE_Y];  //saves on a register?
    __local volatile int ray_count_array[BLOCKSIZE_Y]; //TODO: make a macro
    next_ray_array[y]  = 0;
    ray_count_array[y] = 0;

    kd_stack_elem stack[STACK_SIZE]; //honestly im not sure about the plus three, it seems unnecessary (micro optomization)
    uint stack_length = 0;

    //NOTE: IT WAS CRASHING WHEN THE VECTORS WERENT ALLIGNED!!!!
    kd_44_matrix elem_mask = (kd_44_matrix)(dumb_data);
    __global uint* warp_counter = dumb_data+16;

    //NOTE: ok nvm then
    //prefetch(elem_mask, sizeof(vec4)*sizeof(vec4)); //put into global cache
    //printf("DOES ANYTHING HAPPEN %u %u \n", x, y);


//traversal_stack[STACK_SIZE+0] = get_local_id(0); //saves a register I guess, was recommended in src.
    //traversal_stack[STACK_SIZE+1] = get_local_id(1);

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
    //t_min = t_max = min());//(vec3) SCENE_MIN; //SCENE_MIN is a macro
    //barrier(0);
    while(true)
    {
        uint tidx = x;//stack[STACK_SIZE + 0]; // SINGLE WARPS WORTH OF WORK 0-32
        uint widx = y;//stack[STACK_SIZE + 1]; // WARPS PER SM 0-4 (for example)
        __local volatile int* local_pool_ray_count = ray_count_array+widx;
        __local volatile int* local_pool_next_ray  = next_ray_array+widx;

        //Grab new rays
        if(tidx == 0 && *local_pool_ray_count <= 0) //only the first work item gets memory
        {
            *local_pool_next_ray = atomic_add(warp_counter, LOAD_BALANCER_BATCH_SIZE); //batch complete
            //*local_pool_next_ray -= LOAD_BALANCER_BATCH_SIZE;
// *warp_counter;
            *local_pool_ray_count = LOAD_BALANCER_BATCH_SIZE;
        }
//lol help there are no barriers

        {

            ray_indx = *local_pool_next_ray + tidx;

            if(ray_indx >= num_rays) //ray index is past num rays, work is done
                break;

            if(tidx == 0)
            {
                *local_pool_next_ray += 32;
                *local_pool_ray_count -= 32;
            }

            r = ray_buffer[ray_indx];
            //r.orig = (vec3)(0,0, -2);
            //printf("%f %f %f\n", r.orig.x, r.orig.y, r.orig.z);
            //flaot scene_min_t = min(((vec3)SCENE_MIN - ray.origin) * );

            //assert(ray_indx<1080*1920);

            t_hit = INFINITY; //infinity
            /*printf("BOX: (%f %f %f), (%f %f %f)\n",
                   ((vec3)SCENE_MIN).x, ((vec3)SCENE_MIN).y,
                   ((vec3)SCENE_MIN).z, ((vec3)SCENE_MAX).x,
                   ((vec3)SCENE_MAX).y, ((vec3)SCENE_MAX).z);
                   printf("RAY: (%f %f %f), (%f %f %f)\n", r.orig.x,
                   r.orig.y, r.orig.z, r.dir.x, r.dir.y, r.dir.z);*/
            /*if(hitBoundingBox((vec3) SCENE_MIN, (vec3) SCENE_MAX, r))
            {
                //printf("K COLLISION\n");
            }*/
            if(!getTBoundingBox((vec3) SCENE_MIN, (vec3) SCENE_MAX, r, &scene_t_min, &scene_t_max)) //SCENE_MIN is a macro
            {
                scene_t_max = -INFINITY;//t_max = INFINITY;
                //printf("Shit\n");
            }
            //if(isinf(scene_t_max))
            //printf("Shit2 (%f %f %f) (%f %f %f)\n",
            //r.orig.x, r.orig.y, r.orig.z,
            //r.dir.x,  r.dir.y,  r.dir.z);

            t_max = t_min = scene_t_min;

            stack_length = 0;
            root = *((__global kd_tree_node*) kd_tree);
            //assert(root.left_index == 32);
        }
        stack_length = 0;
        //barrier(0);

        //printf("not bad %u \n", stack_length);
        //if(!(t_max < scene_t_max))
        //printf("REALLY BAD %f < %f\n", t_max, scene_t_max);
        //unsigned int num_times_entered_pdown = 0;
        //unsigned int nodes_traversed = 0;
        while(t_max < scene_t_max)
        {
            //printf("HIT BOUNDING BOX");
            //printf("happened_og %u\n", stack_length);
            if(stack_length == (uint) 0)
            {
                node  = root; //root
                current_type = KDTREE_NODE;
                t_min = t_max;
                t_max = scene_t_max;
                pushdown = true;
                //num_times_entered_pdown++;
            }
            else
            { //pop
                //printf("happened %u\n", stack_length);
                //TODO: update this
                //assert(stack_length == 1);
                //printf("K it went around once!\n");
                t_min = stack[stack_length-1].min;
                t_max = stack[stack_length-1].max;
                kd_update_state(kd_tree, stack[stack_length-1].node, &current_type, &node, &leaf);
                //;current_type = *((__global uchar*)(kd_tree+stack[stack_length-1].node));


                /*if(current_type == KDTREE_NODE)
                {
                    node = *((__global kd_tree_node*)(kd_tree+stack[stack_length-1].node));
                }
                else
                {

                }*/

                //printf("K it retrieved some data from stack!\n");
                stack_length--;
                pushdown = false;
                //printf("K it retrieved all data stack!\n");
            }


            while(current_type != KDTREE_LEAF)
            {
                //printf(":) ok test meme\n");
                //nodes_traversed++;
                //NOTE: none of this branches
                unsigned char k = node.k;
                //assert(k<=2);
                //assert(node.type<=2);
                float t_split = (node.b - get_elem(r.orig, k, elem_mask)) /
                                 get_elem(r.dir, k, elem_mask);
                //printf("THING2: %f %f\n", t_min, t_max);

                //printf("%f %f\n", t_min, t_split);

//if(get_local_id(0)==0)
                //printf("\n%f\n", t_split);
                //NOTE: CRASH HEPPENSE SOMEWHERE HERE
                //NOTE: something weird happens to the indexs of the nodes
                bool left_close =
                    (get_elem(r.orig, k, elem_mask) < node.b) ||
                    (get_elem(r.orig, k, elem_mask) == node.b && get_elem(r.dir, k, elem_mask) <= 0);
                ulong thing = left_close ? 0xffffffffffffffff : 0;
                ulong first  = select(node.right_index, node.left_index,
                                      thing);
                ulong second = select(node.left_index, node.right_index,
                                      thing);
                //printf(":) ok test meme %d %llu %llu %llu %llu\n", (int) node.type, first, second, node.left_index, node.right_index);
                //dbg_print_node(node);
                //assert(first<150000);
                //assert(second<150000);
                //printf("%f %f %f\n", t_split, t_min, t_max);
                if( t_split > t_max || t_split <= 0)  //NOTE: branching necessary
                {
                    kd_update_state(kd_tree, first, &current_type, &node, &leaf);
                    //printf("THING\n");
                }
                else if(t_split < t_min)
                {
                    kd_update_state(kd_tree, second, &current_type, &node, &leaf);
                    //printf("THING2\n");
                }
                else
                {
                    //assert(stack_length!=(ulong)STACK_SIZE-1);

                    //update
                    //printf("SL: %u\n", stack_length);
                    //if(stack_length >= STACK_SIZE)
                    //printf("test meme 1 %llu\n", stack_length);

                    stack[stack_length++] = (kd_stack_elem) {second, t_split, t_max}; //push
                    kd_update_state(kd_tree, first, &current_type, &node, &leaf);
                    //printf("test meme 1 complete\n");

                    t_max = t_split;
                    pushdown = false;
                }

                if(pushdown)
                {
                    root = node;//UPDATE
                }

            }

            //printf("HEY found something %d\n", leaf.num_triangles);
            //Found leaf
            for(ulong t = 0; t <leaf.num_triangles; t++)
            {
//                printf("%u\n", (unsigned int)leaf.num_triangles);
                //assert(leaf.triangle_start-t == 0);
                vec3 tri[4];
                unsigned int index_offset =
                    *(__global uint*)(kd_tree+leaf.triangle_start+(t*sizeof(unsigned int)));
                //get vertex (first element of each index)
                int4 idx_0 = read_imagei(indices, index_offset+0);
                int4 idx_1 = read_imagei(indices, index_offset+1);
                int4 idx_2 = read_imagei(indices, index_offset+2);

                tri[0] = read_imagef(vertices, idx_0.x).xyz;
                tri[1] = read_imagef(vertices, idx_1.x).xyz;
                tri[2] = read_imagef(vertices, idx_2.x).xyz;
                /*printf("%f %f %f : %f %f %f : %f %f %f %llu\n",
                       tri[0].x, tri[0].y, tri[0].z,
                       tri[1].x, tri[1].y, tri[1].z,
                       tri[2].x, tri[2].y, tri[2].z,
                       t);*/

                vec3 hit_coords; // t u v
                if(does_collide_triangle(tri, &hit_coords, r)) //TODO: optomize
                {
                    //printf("COLLISION\n");
                    if(hit_coords.x<=0)
                        continue;
                    if(hit_coords.x < t_hit)
                    {
                        t_hit = hit_coords.x;     //t
                        hit_info = hit_coords.yz; //u v
                        tri_indx = index_offset;
                    }
                    //printf("COLLISION\n");
                    //t_hit = min(t_hit, hit_coords.x); //t  NOTE: this is faster but it saves computation is we store u and v aswell
                    if(t_hit < t_min) // goes by closest to furthest, so if it hits it will be the closest
                    {//early exit
                        //remove that
                        //printf("update\n");
                        scene_t_min = -INFINITY;
                        //break; //TODO: do something NOTE: COULD BE EVEN FASTER WHEN I ACTUALLY FIX THIS
                    }
                }

            }


        }
        //By this point a triangle will have been found.
        kd_tree_collision_result result = {0};

        if(!isinf(t_hit))//if t_hit != INFINITY
        {
            result.triangle_index = tri_indx;
            result.t = t_hit;
            result.u = hit_info.x;
            result.v = hit_info.y;
            //printf("GOOD %d %u\n", ray_indx, (int)tri_indx);
            //printf("Good (%f %f %f) (%f %f %f) %f %u %u\n",
            //r.orig.x, r.orig.y, r.orig.z,
            //r.dir.x,  r.dir.y,  r.dir.z,
            //scene_t_min, num_times_entered_pdown,
            //nodes_traversed);
        }
        else
        {
            //printf("Shit1 (%f %f %f) (%f %f %f) %f %u %u\n",
            //r.orig.x, r.orig.y, r.orig.z,
            //r.dir.x,  r.dir.y,  r.dir.z,
            //scene_t_min, num_times_entered_pdown,
            //nodes_traversed);
            //printf("No Collision (%f, %f, %f) %d %f %f\n",
            //r.orig.x, r.orig.y, r.orig.z, stack_length, t_max, scene_t_max);
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
    //return;
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
    int4 i1 = read_imagei(indices, res.triangle_index);
    int4 i2 = read_imagei(indices, res.triangle_index+1);
    int4 i3 = read_imagei(indices, res.triangle_index+2);
    mesh m = meshes[i1.w];
    material mat = material_buffer[m.material_index];

    vec3 normal =
        read_imagef(normals, i1.y).xyz*(1-res.u-res.v)+
        read_imagef(normals, i2.y).xyz*res.u+
        read_imagef(normals, i3.y).xyz*res.v;

    normal = (normal+1) / 2;

    out_tex[offset] = get_colour( (vec4) (normal,1) );
}
