#pragma once

//typedef struct{} sphere;
struct sphere;

typedef struct
{
    //Primatives

    //Spheres
    sphere* spheres;
    unsigned int num_spheres; //NOTE: must be constant.
    bool spheres_changed; //should re-push values.
    //Planes
    plane* planes;
    unsigned int num_planes; //NOTE: must be constant.
    bool planes_changed; //should re-push values.


} scene;
