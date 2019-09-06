#include <kdtree.h>
#include <scene.h>

#define KDTREE_EPSILON 0.001f

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

//inline doesn't work on osx I gues?
kd_tree_sah_results kd_tree_sah_results_c(float cost, uint8_t side)
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
    uint8_t side;
    float cost;

} kd_tree_find_plane_results;

//inline doesn't work on osx I gues?
bool kd_tree_event_lt(kd_tree_event* left, kd_tree_event* right)
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


#define E KDTREE_EPSILON
//Optional Lambda
float kd_tree_lambda(int NL, int NR, float PL, float PR)
{
    if( (NL <= E || NR <= E) && !(PL >= 1.0f-E || PR >= 1.0f-E) )
        return 0.8f;
    return 1.0f;
}
#undef E

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
    //buffer 1 is guarenteed to be to the direct left of buffer 2
    kd_tree_event_buffer event_out;
    event_out.num_events = buf1.num_events + buf2.num_events;

    event_out.events = (kd_tree_event*) malloc(sizeof(kd_tree_event) * event_out.num_events);


    uint32_t buf1_i, buf2_i, eo_i;
    buf1_i = buf2_i = eo_i = 0;

    while(buf1_i != buf1.num_events || buf2_i != buf2.num_events)
    {
        if(buf1_i == buf1.num_events)
        {
            event_out.events[eo_i++] = buf2.events[buf2_i++];
            continue;
        }

        if(buf2_i == buf2.num_events)
        {
            event_out.events[eo_i++] = buf1.events[buf1_i++];
            continue;
        }

        if( kd_tree_event_lt(buf1.events+buf1_i, buf2.events+buf2_i) )
            event_out.events[eo_i++] = buf1.events[buf1_i++];
        else
            event_out.events[eo_i++] = buf2.events[buf2_i++];
    }
    assert(eo_i == event_out.num_events);
    memcpy(buf1.events, event_out.events, sizeof(kd_tree_event) * event_out.num_events);
    free(event_out.events);
    event_out.events = buf1.events;

    return event_out;
}

kd_tree_event_buffer kd_tree_mergesort_event_buffer(kd_tree_event_buffer buf)
{

    if(buf.num_events == 1)
        return buf;


    int firstHalf = (int)ceil( (float)buf.num_events / 2.f);


    kd_tree_event_buffer buf1 = {buf.events, firstHalf, };
    kd_tree_event_buffer buf2 = {buf.events+firstHalf, buf.num_events-firstHalf};


    buf1 = kd_tree_mergesort_event_buffer(buf1);
    buf2 = kd_tree_mergesort_event_buffer(buf2);


    return kd_tree_merge_event_buffers(buf1, buf2);
}


kd_tree* kd_tree_init()
{
    kd_tree* tree = malloc(sizeof(kd_tree));
    tree->root = NULL;
    //Defaults
    tree->k    = 3;
    tree->max_recurse = 50;
    tree->tri_for_leaf_threshold = 2;
    tree->num_nodes_total     = 0;
    tree->num_tris_padded     = 0;
    tree->num_traversal_nodes = 0;
    tree->num_leaves          = 0;
    tree->num_indices_total   = 0;
    tree->buffer_size         = 0;
    tree->buffer              = NULL;
    tree->cl_kd_tree_buffer   = NULL;
    xv3_zero(tree->bounds.min);
    xv3_zero(tree->bounds.max);
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



kd_tree_find_plane_results kd_tree_find_plane(kd_tree* tree, AABB V,
                                              kd_tree_triangle_buffer tri_buf)
{
    float     best_cost = INFINITY;
     kd_tree_find_plane_results result;


    for(int k = 0; k < tree->k; k++)
    {
        kd_tree_event_buffer event_buf = {NULL, 0}; //gets rid of an initialization warning I guess?
        {// Generate events
            //Divide by three because we only want tris
            event_buf.num_events = tri_buf.num_triangles*2;

            event_buf.events = malloc(sizeof(kd_tree_event)*event_buf.num_events);
            unsigned int j = 0;
            for (int i = 0; i < tri_buf.num_triangles; i++)
            {
                AABB tv, B;
                AABB_construct_from_triangle(&tv, tree->s->mesh_indices+tri_buf.triangle_buffer[i],
                                             tree->s->mesh_verts);
                AABB_clip(&B, &tv, &V);
                if(AABB_is_planar(&B, k))
                {
                    event_buf.events[j++] = (kd_tree_event) {i*3, B.min[k], k, KDTREE_PLANAR};
                }
                else
                {
                    event_buf.events[j++] = (kd_tree_event) {i*3, B.min[k], k, KDTREE_START};
                    event_buf.events[j++] = (kd_tree_event) {i*3, B.max[k], k, KDTREE_END};
                }
            }
			event_buf.num_events = j;

            int last_num_events = event_buf.num_events;
            event_buf = kd_tree_mergesort_event_buffer(event_buf);
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
            while(i < event_buf.num_events && event_buf.events[i].b == p.b && event_buf.events[i].type == KDTREE_END)
            {
                Pe += 1;
                i++;
            }
            while(i < event_buf.num_events && event_buf.events[i].b == p.b && event_buf.events[i].type == KDTREE_PLANAR)
            {
                Pp += 1;
                i++;
            }
            while(i < event_buf.num_events && event_buf.events[i].b == p.b && event_buf.events[i].type == KDTREE_START)
            {
                Ps += 1;
                i++;
            }

            NP =  Pp;
            NR -= Pp;
            NR -= Pe;

            kd_tree_sah_results results = kd_tree_SAH(k, AABB_ilerp(V, k, p.b), V, NL, NR, NP);

            if (results.cost < best_cost)
            {
                best_cost = results.cost;
                result.p = p;
                result.side = results.side;

                result.NL = NL;
                result.NR = NR;
                result.NP = NP;
                result.cost = results.cost; //just the min cost, really confusing syntax
            }

            NL += Ps;
            NL += NP;
            NP =  0;

        }
        free(event_buf.events);
    }

    return result;
}

void kd_tree_classify(kd_tree* tree, kd_tree_triangle_buffer tri_buf,
                      kd_tree_find_plane_results results,
                      kd_tree_triangle_buffer* TL_out, kd_tree_triangle_buffer* TR_out)
{
    kd_tree_triangle_buffer TL;
    kd_tree_triangle_buffer TR;
    TL.num_triangles   = results.NL + (results.side == KDTREE_LEFT ? results.NP : 0);
    TL.triangle_buffer = (unsigned int*) malloc(sizeof(unsigned int)*TL.num_triangles); //NOTE: memory leak, never freed.
    TR.num_triangles   = results.NR + (results.side == KDTREE_RIGHT ? results.NP : 0);
    TR.triangle_buffer = (unsigned int*) malloc(sizeof(unsigned int)*TR.num_triangles);
    unsigned int TLI, TRI;
    TLI = TRI = 0;
    for(int i = 0; i < tri_buf.num_triangles; i++)
    {
        bool isLeft = false;
        bool isRight = false;
        for(int j = 0; j < 3; j++)
        {

            float p = tree->s->mesh_verts
                  [ tree->s->mesh_indices
                  [ tri_buf.triangle_buffer[i]+j ][0] ][results.p.k];
            if(p < results.p.b)
                isLeft = true;
            if(p > results.p.b)
                isRight = true;
        }

        //Favour the right rn
        if(isLeft && isRight) //should be splitting.
        {
            TR.triangle_buffer[TRI++] = tri_buf.triangle_buffer[i];
            TL.triangle_buffer[TLI++] = tri_buf.triangle_buffer[i];
        }
        else if(!isLeft && !isRight)
        {
            if(results.side == KDTREE_LEFT)
                TL.triangle_buffer[TLI++] = tri_buf.triangle_buffer[i];
            else if(results.side == KDTREE_RIGHT)
                TR.triangle_buffer[TRI++] = tri_buf.triangle_buffer[i];
            else
            {//implement this
                printf("really bad\n");
                assert(1!=1);
            }
        }
        else if(isLeft)
            TL.triangle_buffer[TLI++] = tri_buf.triangle_buffer[i];
        else if(isRight)
            TR.triangle_buffer[TRI++] = tri_buf.triangle_buffer[i];
    }
    *TL_out = TL;
    *TR_out = TR;

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

    return false;
}

kd_tree_node* kd_tree_construct_rec(kd_tree* tree, AABB V, kd_tree_triangle_buffer tri_buf,
                                    unsigned int depth)
{
    kd_tree_node* node = kd_tree_node_init();

    tree->num_nodes_total++;
    if(kd_tree_should_terminate(tree, tri_buf.num_triangles, V, depth))
    {
        node->triangles = tri_buf;
        tree->num_leaves++;
        tree->num_indices_total += tri_buf.num_triangles;
        tree->num_tris_padded   += tri_buf.num_triangles % 8;
        return node;
    }

    kd_tree_find_plane_results res = kd_tree_find_plane(tree, V, tri_buf);

	if(res.cost > KDTREE_KI*(float)tri_buf.num_triangles)
    {
        node->triangles = tri_buf;
        tree->num_leaves++;
        tree->num_indices_total += tri_buf.num_triangles;
        tree->num_tris_padded   += tri_buf.num_triangles % 8;

        return node;
    }


    tree->num_traversal_nodes++;


    uint8_t     k = res.p.k;
    float world_b = res.p.b;

    node->k = k;
    node->b = world_b; //local b is honestly useless

    assert(node->b != V.min[k]);
    assert(node->b != V.max[k]);

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
	assert(tree->s->num_mesh_indices % 3 == 0);
    kd_tree_triangle_buffer buf;
    buf.num_triangles   = tree->s->num_mesh_indices/3;
    buf.triangle_buffer = (unsigned int*) malloc(sizeof(unsigned int) * buf.num_triangles);

	for (int i = 0; i < buf.num_triangles; i++)
		buf.triangle_buffer[i] = i * 3;

    return buf;
}

void kd_tree_construct(kd_tree* tree) //O(n log^2 n) implementation
{
    assert(tree->s != NULL);

    if(tree->s->num_mesh_indices == 0)
    {
        printf("WARNING: Skipping k-d tree Construction, num_mesh_indices is 0.\n");
        return;
    }

    AABB V;
    AABB_construct_from_vertices(&V, tree->s->mesh_verts, tree->s->num_mesh_verts); //works
    printf("DBG: kd-tree volume: (%f, %f, %f)  (%f, %f, %f)\n", V.min[0], V.min[1], V.min[2], V.max[0], V.max[1], V.max[2]);

    tree->bounds = V;

    tree->root = kd_tree_construct_rec(tree, V, kd_tree_gen_initial_tri_buf(tree), 0);
}

unsigned int _kd_tree_write_buf(char* buffer, unsigned int offset,
                                                   void* data, size_t size)
{
    memcpy(buffer+offset, data, size);
    return offset + size;
}

//returns finishing offset
unsigned int kd_tree_generate_serialized_buf_rec(kd_tree* tree, kd_tree_node* node, unsigned int offset)
{
    //NOTE: this could really just be two functions
    if(kd_tree_node_is_leaf(node)) // leaf
    {

        { //leaf body
            _skd_tree_leaf_node l;
            l.type = KDTREE_LEAF;
            l.num_triangles = node->triangles.num_triangles;
            //printf("TEST %u \n", l.num_triangles);
            //assert(l.num_triangles != 0);
            offset = _kd_tree_write_buf(tree->buffer, offset, &l, sizeof(_skd_tree_leaf_node));
        }

        for(int i = 0; i < node->triangles.num_triangles; i++) //triangle indices
        {
            offset = _kd_tree_write_buf(tree->buffer, offset,
                                        node->triangles.triangle_buffer+i, sizeof(unsigned int));
        }
        if(node->triangles.num_triangles % 2)
            offset += 4;//if it isn't alligned with a long add 4 bytes (8 byte allignment)

        return offset;
    }
    else // traversal node
    {
        _skd_tree_traversal_node n;
        n.type = KDTREE_NODE;
        n.k = node->k;
        n.b = node->b;
        unsigned int struct_start_offset = offset;
        offset += sizeof(_skd_tree_traversal_node);

        unsigned int left_offset  = kd_tree_generate_serialized_buf_rec(tree, node->left, offset);
        //this goes after the left node
        unsigned int right_offset = kd_tree_generate_serialized_buf_rec(tree, node->right, left_offset);

        n.left_ind  = offset/8;
        n.right_ind = left_offset/8;

        memcpy(tree->buffer+struct_start_offset, &n, sizeof(_skd_tree_traversal_node));

        return right_offset;
    }
}

void kd_tree_generate_serialized(kd_tree* tree)
{
    if(tree->s->num_mesh_indices == 0)
    {
        printf("WARNING: Skipping k-d tree Serialization, num_mesh_indices is 0.\n");
        tree->buffer_size = 0;
        tree->buffer = malloc(1);
        return;
    }

    unsigned int mem_needed = 0;

    mem_needed += tree->num_traversal_nodes * sizeof(_skd_tree_traversal_node); //traversal nodes
    mem_needed += tree->num_leaves * sizeof(_skd_tree_leaf_node); //leaf nodes
    mem_needed += (tree->num_indices_total+tree->num_tris_padded) * sizeof(unsigned int); //triangle indices

    //char* name = malloc(256);
    //sprintf(name, "%d.bkdt", mem_needed);

    tree->buffer_size = mem_needed;
    printf("k-d tree is %d bytes long...", mem_needed);

    tree->buffer = malloc(mem_needed);


    /*FILE* f = fopen(name, "r");
    if(f!=NULL)
    {
        printf("Using cached kd tree.\n");
        fread(tree->buffer, 1, mem_needed, f);
        fclose(f);
    }
    else*/
    kd_tree_generate_serialized_buf_rec(tree, tree->root, 0);

        /*{
        f = fopen(name, "w");
        fwrite(tree->buffer, 1, mem_needed, f);
        fclose(f);
    }
    free(name);*/
}
