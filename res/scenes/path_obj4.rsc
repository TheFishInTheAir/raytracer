{
    "version":
    {
        "major":0,
        "minor":1,
        "type":"DEV"
    },
    "name":"Path Tracing Test Scene",


    "materials":
    [
        {
            "r":0.7,
            "g":0.7,
            "b":0.75,
            "reflectivity":0.7
        },
        {
            "r":0.7,
            "g":0.7,
            "b":0.75,
            "reflectivity":1
        },
        {
            "r":0.73,
            "g":0.9411,
            "b":0.819,
            "reflectivity":0.05
        },
        {
            "r":0.8588,
            "g":0.352,
            "b":0.2588,
            "reflectivity":0.2
        },
        {
            "r":1,
            "g":1,
            "b":1,
            "reflectivity":0.1
        }
    ],

    "meshes":
    [
        {
            "url":"mesh/mirror-edge.obj",
            "mat_index": 3,
            "px":0,
            "py":0,
            "pz":0,
            "sx":1,
            "sy":1,
            "sz":1,

            "___COMMENT___":"need to add rotation"
        },
        {
            "url":"mesh/mirror_edge_lights.obj",
            "mat_index": 1,
            "px":0,
            "py":0,
            "pz":0,
            "sx":1,
            "sy":1,
            "sz":1,

            "___COMMENT___":"need to add rotation"
        }
    ],

    "primitives":
    {
        "spheres":
        [
            {
                "x":-0.4,
                "y":-120301237,
                "z":-2,
                "radius":4,
                "mat_index":1
            }
        ],
        "planes":
        [
            {
                "x":0,
                "y":-10.01,
                "z":0,
                "nx":0,
                "ny":1,
                "nz":0,

                "mat_index":2
            }
        ]
    }
}
