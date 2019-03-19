#pragma once
#include <stdint.h>
#include <stdbool.h>

struct scene;

//TODO: make these variable from the ui, eventually
#define KDTREE_KT 1.0f //Cost for traversal
#define KDTREE_KI 1.5f //Cost for intersection


typedef struct kd_tree_triangle_buffer
{
    unsigned int* triangle_buffer;
    unsigned int  num_triangles;
} kd_tree_triangle_buffer;


//NOTE: should the voxel be stored in here?
//NOTE: should the depth be stored in here?
typedef struct kd_tree_node
{
    uint8_t k; //Splittign Axis                  Bytes: 1, 1
    float   b; //Local Split Ratio               Bytes: 4, 5

    struct kd_tree_node* left; //                       Bytes: 8, 13  NOTE: this could prob be stored as uint
    struct kd_tree_node* right;//                       Bytes: 8, 21

    kd_tree_triangle_buffer triangles;

} kd_tree_node;

typedef struct kd_tree
{
    kd_tree_node* root;
    unsigned int k; //Num dimensions, should always be three in this case

    unsigned int max_recurse;
    unsigned int tri_for_leaf_threshold;

    scene* s;
    //AABB V; //Total bounding box

} kd_tree;


kd_tree*      kd_tree_init();
kd_tree_node* kd_tree_node_init();

bool kd_tree_node_is_leaf(kd_tree_node*);
void kd_tree_construct(kd_tree* tree); //O(n log^2 n) implementation
