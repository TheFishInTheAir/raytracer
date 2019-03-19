#include <kdtree.h>
#include <scene.h>

#define KDTREE_EPSILON 0.0001f

#define KDTREE_BOTH  0
#define KDTREE_LEFT  1
#define KDTREE_RIGHT 2

#define KDTREE_END   0
#define KDTREE_PLANAR 1
#define KDTREE_START  2

//Literally an index buffer to the index buffer

typedef struct kd_tree_event
{
    unsigned int tri_index_offset;
    float   b;
    uint8_t k;
    uint8_t type;
} kd_tree_event;

typedef struct kd_tree_sah_results
{
    float cost;
    uint8_t side; //1 left, 2 right
} kd_tree_sah_results;

inline kd_tree_sah_results kd_tree_sah_results_c(float cost, uint8_t side)
{
    kd_tree_sah_results r;
    r.cost = cost;
    r.side = side;
    return r;
}

typedef struct kd_tree_find_plane_results
{
    kd_tree_event p;
    unsigned int NL;
    unsigned int NR;
    unsigned int NP;

} kd_tree_find_plane_results;


inline bool kd_tree_event_lt(kd_tree_event* left, kd_tree_event* right)
{
    return
        (left->b <  right->b)                             ||
        (left->b == right->b && left->type < right->type) ||
        (left->k >  right->k);
}

typedef struct kd_tree_event_buffer
{
    kd_tree_event* events;
    unsigned int  num_events;
} kd_tree_event_buffer;



//Optional Lambda
float kd_tree_lambda(int NL, int NR, float PL, float PR)
{
    if( (NL == 0 || NR == 0) && !(PL == 1.0f || PR == 1.0f) ) //TODO: be less exact for pl pr check, add epsilon
        return 0.8f;
    return 1.0f;
}

//Cost function
float kd_tree_C(float PL, float PR, uint32_t NL, uint32_t NR)
{
    return kd_tree_lambda(NL, NR, PL, PR) *(KDTREE_KT + KDTREE_KI*(PL*NL + PR*NR));
}

kd_tree_sah_results kd_tree_SAH(uint8_t k, float b, AABB V, int NL, int NR, int NP)
{
    AABB VL;
    AABB VR;
    AABB_divide(V, k, b, &VL, &VR);
    float PL = AABB_surface_area(VL) / AABB_surface_area(V);
    float PR = AABB_surface_area(VR) / AABB_surface_area(V);

    if (PL >= 1-KDTREE_EPSILON || PR >= 1-KDTREE_EPSILON) //NOTE: doesn't look like it but potential source of issues
        return kd_tree_sah_results_c(1000000000.0f, 0);

    float CPL = kd_tree_C(PL, PR, NL+NP, NR);
    float CPR = kd_tree_C(PL, PR, NL, NR+NP);


    if(CPL < CPR)
        return kd_tree_sah_results_c(CPL, KDTREE_LEFT);
    else
        return kd_tree_sah_results_c(CPR, KDTREE_RIGHT);
}


kd_tree_event_buffer kd_tree_merge_event_buffers(kd_tree_event_buffer buf1, kd_tree_event_buffer buf2)
{
    kd_tree_event_buffer event_out;
    event_out.events = (kd_tree_event*)
        malloc(sizeof(kd_tree_event) * (buf1.num_events + buf2.num_events));

    uint32_t buf1_i, buf2_i, eo_i;
    buf1_i = buf2_i = eo_i = 0;

    while(buf1_i != buf1.num_events-1 || buf2_i != buf2.num_events-1)
    {
        if(buf1_i != buf1.num_events-1)
        {
            event_out.events[eo_i++] = buf2.events[buf2_i++];
            continue;
        }

        if(buf2_i != buf2.num_events-1)
        {
            event_out.events[eo_i++] = buf1.events[buf1_i++];
            continue;
        }

        if( kd_tree_event_lt(buf1.events+buf1_i, buf2.events+buf2_i) )
            event_out.events[eo_i++] = buf1.events[buf1_i];
        else
            event_out.events[eo_i++] = buf2.events[buf2_i];
    }

    return event_out;
}

kd_tree_event_buffer kd_tree_mergesort_event_buffer(kd_tree_event_buffer buf)
{
    if(buf.num_events == 1)
        return buf;

    int firstHalf = (int)ceil( (float)buf.num_events / 2.f);

    kd_tree_event_buffer buf1 = {buf.events, firstHalf};
    kd_tree_event_buffer buf2 = {buf.events+firstHalf, buf.num_events-firstHalf};

    buf1 = kd_tree_mergesort_event_buffer(buf1);
    buf2 = kd_tree_mergesort_event_buffer(buf2);

    return kd_tree_merge_event_buffers(buf1, buf2);
}


kd_tree* kd_tree_init()
{
    kd_tree* tree = malloc(sizeof(kd_tree));
    tree->root = NULL;
    tree->k    = 3; //Default

    return tree;
}

kd_tree_node* kd_tree_node_init()
{
    kd_tree_node* node = malloc(sizeof(kd_tree_node));
    node->k = 0;
    node->b = 0.5f; //generic default, shouldn't matter with SAH anyways

    node->left  = NULL;
    node->right = NULL;

    return node;
}

bool kd_tree_node_is_leaf(kd_tree_node* node)
{
    if(node->left == NULL || node->right == NULL)
    {
        assert(node->left == NULL && node->right == NULL);
        return true;
    }

    return false;
}



kd_tree_find_plane_results kd_tree_find_plane(kd_tree* tree, AABB V,// ivec3* index_buffer,
                                              kd_tree_triangle_buffer tri_buf)
{
    float     best_cost = 10000000000.0f; //TODO: replace with infinity macro
    //uint8_t   best_side = 0; //TODO: use this
    //kd_tree_event best_p = NULL;
    kd_tree_find_plane_results result;


    for(int k = 0; k < tree->k; k++)
    {
        kd_tree_event_buffer event_buf = {NULL, 0}; //gets rid of an initialization warning I guess?


        {// Generate events
            //Divide by three because we only want tris
            event_buf.events = malloc(sizeof(kd_tree_event)*(tri_buf.num_triangles)*2);
            unsigned int j = 0;
            for (int i = 0; i < tri_buf.num_triangles; i++)
            {
                AABB tv, B;
                AABB_construct_from_triangle(&tv, tree->s->mesh_indices+tri_buf.triangle_buffer[i],
                                             tree->s->mesh_verts);
                AABB_clip(&B, &tv, &V);

                event_buf.events[j++] = (kd_tree_event) {i*3, B.min[k]-KDTREE_EPSILON, k, KDTREE_START};
                event_buf.events[j++] = (kd_tree_event) {i*3, B.max[k]+KDTREE_EPSILON, k, KDTREE_END};
            }
            int last_num_events = event_buf.num_events;
            event_buf = kd_tree_mergesort_event_buffer(event_buf); //
            assert(event_buf.num_events == last_num_events);
        }

        int NL, NP, NR;
        NL = NP = 0;
        NR = tri_buf.num_triangles;
        for (int i = 0; i < event_buf.num_events;)
        {
            kd_tree_event p = event_buf.events[i];
            int Ps, Pe, Pp;
            Ps = Pe = Pp = 0;

            while(i < event_buf.num_events && event_buf.events[i].b == p.b)
            {
                if (event_buf.events[i].type == KDTREE_END)
                    Pe += 1;
                else if(event_buf.events[i].type == KDTREE_START)
                    Ps += 1;
                i++;
            }

            NR -= Pe;

            kd_tree_sah_results results = kd_tree_SAH(k, AABB_ilerp(V, k, p.b), V, NL, NR, 0);

            if (results.cost < best_cost)
            {
                best_cost = results.cost;
                result.p = p;
                //NOTE: Favour the right, its not ideal, but NP isn't working right now
                result.NL = tri_buf.num_triangles-NR;
                result.NR = NR;
                result.NP = NP;
            }

            NL += Ps;

            NP = 0; //TODO: do the stuff for planar tris, also use sides
        }
    }
    return result;
}

void kd_tree_classify(kd_tree* tree, kd_tree_triangle_buffer tri_buf,
                      kd_tree_find_plane_results results,
                      kd_tree_triangle_buffer* TL_out, kd_tree_triangle_buffer* TR_out)
{
    kd_tree_triangle_buffer TL;
    kd_tree_triangle_buffer TR;
    TL.num_triangles   = results.NL;
    TL.triangle_buffer = (unsigned int*) malloc(sizeof(unsigned int)*results.NL);
    TR.num_triangles   = results.NR;
    TR.triangle_buffer = (unsigned int*) malloc(sizeof(unsigned int)*results.NR);
    unsigned int TLI, TRI;
    TLI = TRI = 0;
    for(int i = 0; i < tri_buf.num_triangles; i++)
    {
        bool isLeft = false;
        bool isRight = false;
        for(int j = 0; j < 3; j++)
        {
            for(int k = 0; k < 3; k++)
            {
                //uint tri_start = tri_buf.triangle_buffer[i]
                //uint vert_indx = tree->s->mesh_indices[tri_start+j][0]
                //vec3 vert      = tree->s->mesh_vertices[vert_indx]
                //float p        = vert[k]

                float p = tree->s->mesh_verts
                        [ tree->s->mesh_indices
                        [ tri_buf.triangle_buffer[i]+j ][0] ][k];

                if (p < results.p.b)
                    isLeft = true;
                else if(p > results.p.b)
                    isRight = true;
                //have an else for on the line
            }
        }

        //Favour the right rn
        if((isLeft && isRight) || !(isLeft && isRight))
            TR.triangle_buffer[TRI++] = tri_buf.triangle_buffer[i];
        else if(isLeft)
            TL.triangle_buffer[TLI++] = tri_buf.triangle_buffer[i];
        else if(isRight)
            TR.triangle_buffer[TRI++] = tri_buf.triangle_buffer[i];
    }
    *TL_out = TL;
    *TR_out = TR;
    //kd_tree_triangle_buffer TP;

}

bool kd_tree_should_terminate(kd_tree* tree, unsigned int num_tris, AABB V, unsigned int depth)
{
    for(int k = 0; k < tree->k; k++)
        if(AABB_is_planar(&V, k))
            return true;
    if(depth == tree->max_recurse)
        return true;
    if(num_tris <= tree->tri_for_leaf_threshold)
        return true;

    //TODO: also add the case in which cost of dividing more exceeds cost of just makign a leaf.

    return false;
}

kd_tree_node* kd_tree_construct_rec(kd_tree* tree, AABB V, kd_tree_triangle_buffer tri_buf,
                                    unsigned int depth)
{
    kd_tree_node* node = kd_tree_node_init();

    if(kd_tree_should_terminate(tree, tri_buf.num_triangles, V, depth))
    {
        node->triangles = tri_buf;
        return node;
    }

    kd_tree_find_plane_results res = kd_tree_find_plane(tree, V, tri_buf);

    uint8_t     k = res.p.k;
    float world_b = res.p.b;

    AABB VL;
    AABB VR;
    AABB_divide_world(V, k, world_b, &VL, &VR);

    kd_tree_triangle_buffer TL, TR;
    kd_tree_classify(tree, tri_buf, res, &TL, &TR);

    node->left  = kd_tree_construct_rec(tree, VL, TL, depth+1);
    node->right = kd_tree_construct_rec(tree, VR, TR, depth+1);

    return node;
}

kd_tree_triangle_buffer kd_tree_gen_initial_tri_buf(kd_tree* tree)
{
    kd_tree_triangle_buffer buf;
    buf.num_triangles   = tree->s->num_mesh_indices/3;
    buf.triangle_buffer = (unsigned int*) malloc(sizeof(unsigned int) * buf.num_triangles);

    unsigned int j = 0;
    for(int i = 0; i < buf.num_triangles; i++)
        buf.triangle_buffer[j++] = i*3;

    return buf;
}

void kd_tree_construct(kd_tree* tree) //O(n log^2 n) implementation
{
    assert(tree->s != NULL);

    AABB V; //TODO: create it
    AABB_construct_from_vertices(&V, tree->s->mesh_verts, tree->s->num_mesh_verts);
    tree->root = kd_tree_construct_rec(tree, V, kd_tree_gen_initial_tri_buf(tree), 0);
}
