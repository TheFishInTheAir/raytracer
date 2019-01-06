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
            "url":"mesh/test.obj",
            "mat_index": 3,
            "px":-4,
            "py":2,
            "pz":-5,
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
                "y":7,
                "z":-2,
                "radius":4,
                "mat_index":1
            }
        ],
        "planes":
        [
            {
                "x":0,
                "y":-1,
                "z":0,
                "nx":0,
                "ny":1,
                "nz":0,

                "mat_index":2
            },
            {
                "x":0,
                "y":0,
                "z":-10,
                "nx":0,
                "ny":0,
                "nz":1,

                "mat_index":2
            }
        ]
    }
}
