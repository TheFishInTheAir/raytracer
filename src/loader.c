#include <loader.h>
#include <parson.h>
#include <vec.h>
#include <tinyobj_loader_c.h>

#ifndef WIN32
#include <libproc.h>
#include <unistd.h>

#define FILE_SEP '/'

char* _get_os_pid_bin_path()
{
    static bool initialised = false;
    static char path[PROC_PIDPATHINFO_MAXSIZE];
    if(!initialised)
    {
        int ret;
        pid_t pid;
        char path[PROC_PIDPATHINFO_MAXSIZE];

        pid = getpid();
        ret = proc_pidpath(pid, path, sizeof(path));

        if(ret <= 0)
        {
            printf("Error: couldn't get bin path.");
            exit(1);
        }
    }
    return path;
}
#else
#include <windows.h>
#define FILE_SEP '\\'

char* _get_os_pid_bin_path()
{
    static bool initialised = false;
    static char path[260];
    if(!initialised)
    {
        HMODULE hModule = GetModuleHandleW(NULL);

        WCHAR tpath[260];
        GetModuleFileNameW(hModule, tpath, 260);

        char DefChar = ' ';
        WideCharToMultiByte(CP_ACP, 0, tpath, -1, path, 260, &DefChar, NULL);
    }
	return path;
}
#endif

char* load_file(char* url, long *ret_length)
{
    char real_url[260];
    sprintf(real_url, "%s%c%s", _get_os_pid_bin_path(), FILE_SEP, url);

    char * buffer = 0;
    long length;
    FILE * f = fopen (real_url, "rb");

    if (f)
    {
        fseek (f, 0, SEEK_END);
        length = ftell (f);
        fseek (f, 0, SEEK_SET);
        buffer = malloc (length);
        if (buffer)
        {
            fread (buffer, 1, length, f);
        }
        fclose (f);
    }
    if (buffer)
    {
        buffer[length] = '\0';

        load_program_raw(ctx, buffer, kernels, num_kernels, program,
                         macros, num_macros);
    }
    *ret_length = length;
    return buffer;
}




void load_obj(char* data, long data_len) //TODO: READ OVER DEMO AND FINISH
{
    tinyobj_attrib_t attrib;
    tinyobj_shape_t* shapes = NULL;
    size_t num_shapes;
    tinyobj_material_t* materials = NULL; //NOTE: UNUSED
    size_t num_materials;                 //NOTE: UNUSED

    printf("filesize: %d\n", (int)data_len);

    {
        unsigned int flags = TINYOBJ_FLAG_TRIANGULATE;
        int ret = tinyobj_parse_obj(&attrib, &shapes, &num_shapes, NULL,
                                    &num_materials, data, data_len, flags);
        if (ret != TINYOBJ_SUCCESS) {
            return 0;
        }
    }

}

scene load_scene_json(char* json)
{
    printf("Beginning scene loading...")
    scene out_scene;
    JSON_Object *root_object;

    root_object = json_value_get_object(json_parse_string(json));


    //Name
    {
        char* name = json_object_get_string(root_object, "name");
        printf("Scene name: %s", name);
    }

    //Version
    {//TODO: do something with this.
        int major  = (int)json_object_dotget_number(root_object, "version.major");
        int minor  = (int)json_object_dotget_number(root_object, "version.major");
        char* type =      json_object_dotget_string(root_object, "version.type");
    }

    //Materials
    {
        JSON_Array material_array = json_object_get_array("materials");
        out_scene.num_materials   = json_array_get_count(material_array);
        out_scene.materials = (material*) malloc(out_scene.num_materials*sizeof(material));

        for(int i = 0; i < out_scene.num_material; i++)
        {
            JSON_Object mat = json_array_get_object(material_array, i);
            xv_x(out_scene.materials[i].colour) = json_object_get_number(mat, "r");
            xv_y(out_scene.materials[i].colour) = json_object_get_number(mat, "g");
            xv_z(out_scene.materials[i].colour) = json_object_get_number(mat, "b");
            out_scene.materials[i].reflectivity = json_object_get_number(mat, "reflectivity");
        }
    }

    //Materials
    {
        JSON_Array material_array = json_object_get_array("materials");
        out_scene.num_materials   = json_array_get_count(material_array);
        out_scene.materials = (material*) malloc(out_scene.num_materials*sizeof(material));

        for(int i = 0; i < out_scene.num_material; i++)
        {
            JSON_Object mat = json_array_get_object(material_array, i);
            xv_x(out_scene.materials[i].colour) = json_object_get_number(mat, "r");
            xv_y(out_scene.materials[i].colour) = json_object_get_number(mat, "g");
            xv_z(out_scene.materials[i].colour) = json_object_get_number(mat, "b");
            out_scene.materials[i].reflectivity = json_object_get_number(mat, "reflectivity");
        }
    }

    //Meshes
    {
        JSON_Array mesh_array = json_object_get_array("meshes");

        int num_meshes = json_array_get_count(material_array);

        //out_scene.num_materials   = json_array_get_count(material_array);
        //out_scene.materials = (material*) malloc(out_scene.num_materials*sizeof(material));

        for(int i = 0; i < num_meshes; i++)
        {
            JSON_Object mesh = json_array_get_object(mesh_array, i);
            char* url = json_object_get_string(mesh, "url");
            long length;
            char* data = load_file(url, &length);
            load_obj(data, length);
        }
    }

}


scene load_scene_json_url(char* url)
{
    //TODO: load url
    printf("load_scene_json_url is incomplete!");
    exit(1);

    _get_os_pid_bin_path();

    return load_scene_json(""); //TODO: put data
}
