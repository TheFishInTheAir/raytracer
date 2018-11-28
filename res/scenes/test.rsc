{
    "version":
    {
        "major":0,
        "minor":1,
        "type":"DEV"
    },
    "name":"test scene",


    "materials":
    [
        {
            "r":0.8,
            "g":0.7,
            "b":0.2,
            "reflectivity":0.3
        },
        {
            "r":0.5,
            "g":0.3,
            "b":0.6,
            "reflectivity":0.4
        },
        {
            "r":1,
            "g":1,
            "b":1,
            "reflectivity":1
        }
    ],

    "meshes":
    [
        {
            "url":"mesh/norway.obj",
            "px":-4,
            "py":2,
            "pz":-5,
            "sx":1,
            "sy":1,
            "sz":1,

            "___COMMENT___":"need to add rotation,"
        }
    ],

    "primitives":
    {
        "spheres":
        [
            {
                "x":3,
                "y":0.6,
                "z":-7.8,
                "radius":0.5,
                "mat_index":0
            },
            {
                "x":6,
                "y":7,
                "z":-10.8,
                "radius":0.5,
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
            }
        ],
    }
}
