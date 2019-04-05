#pragma once
#include <stdint.h>
#include <stdbool.h>

struct scene;
//struct AABB;
//TODO: make these variable from the ui, eventually
#define KDTREE_KT 1.0f //Cost for traversal
#define KDTREE_KI 1.5f //Cost for intersection

#define KDTREE_LEAF 1
#define KDTREE_NODE 2


//serializable kd traversal node
typedef struct W_ALIGN(16) _skd_tree_traversal_node
{
    uint8_t type;
    uint8_t k;
    float   b;

    size_t left_ind;   //NOTE: always going to be aligned by at least 8 (could multiply by 8 on gpu)
    size_t right_ind;
} U_ALIGN(16) _skd_tree_traversal_node;


//serializable kd leaf node
typedef struct  W_ALIGN(16) _skd_tree_leaf_node
{
    uint8_t type;
    unsigned int num_triangles;
    //uint tri 1
    //uint tri 2
    //uint etc...
} U_ALIGN(16) _skd_tree_leaf_node;

typedef struct kd_tree_triangle_buffer
{
    unsigned int* triangle_buffer;
    unsigned int  num_triangles;
} kd_tree_triangle_buffer;

//NOTE: not using a vec3 for the floats because it would be a waste of space.
typedef struct kd_tree_collision_result
{
    unsigned int triangle_index;
    float t;
    float u;
    float v;
} kd_tree_collision_result;

//NOTE: should the depth be stored in here?
typedef struct kd_tree_node
{
    uint8_t k; //Splittign Axis
    float   b; //World Split plane

    struct kd_tree_node* left;
    struct kd_tree_node* right;

    kd_tree_triangle_buffer triangles;

} kd_tree_node;

typedef struct kd_tree
{
    kd_tree_node* root;
    unsigned int k; //Num dimensions, should always be three in this case



    unsigned int num_nodes_total;
    unsigned int num_tris_padded;
    unsigned int num_traversal_nodes;
    unsigned int num_leaves;
    unsigned int num_indices_total;

    unsigned int max_recurse;
    unsigned int tri_for_leaf_threshold;

    scene* s;
    AABB bounds;

    //Serialized form.
    char* buffer;
    unsigned int buffer_size;
    cl_mem cl_kd_tree_buffer;

    //AABB V; //Total bounding box

} kd_tree;


kd_tree*      kd_tree_init();
kd_tree_node* kd_tree_node_init();

bool kd_tree_node_is_leaf(kd_tree_node*);
void kd_tree_construct(kd_tree* tree); //O(n log^2 n) implementation
void kd_tree_generate_serialized(kd_tree* tree);
