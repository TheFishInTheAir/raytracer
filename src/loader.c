#include <loader.h>
#include <parson.h>
#include <vec.h>
#include <float.h>
#include <tinyobj_loader_c.h>
#include <assert.h>



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
            printf("Error: couldn't get bin path.\n");
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

        *(strrchr(path, FILE_SEP)) = '\0'; //get last occurence;

    }
	return path;
}
#endif

char* load_file(char* url, long *ret_length)
{
    char real_url[260];
    sprintf(real_url, "%s%cres%c%s", _get_os_pid_bin_path(), FILE_SEP, FILE_SEP, url);

    char * buffer = 0;
    long length;
    FILE * f = fopen (real_url, "rb");
    //printf("TEST THING: '%s'.\n", real_url); TODO: Remove

    if (f)
    {
        fseek (f, 0, SEEK_END);
        length = ftell (f)+1;
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

        *ret_length = length;
        return buffer;
    }
    else
    {
        printf("Error: Couldn't load file '%s'.\n", real_url);
        exit(1);
    }
}


//Linked List for Mesh loading
struct obj_list_elem
{
    struct obj_list_elem* next;
    tinyobj_attrib_t attrib;
    tinyobj_shape_t* shapes;
    size_t num_shapes;
    int mat_index;
    mat4 model_mat;
};

void obj_pre_load(char* data, long data_len, struct obj_list_elem* elem,
                  int* num_meshes, unsigned int* num_indices, unsigned int* num_vertices,
                  unsigned int* num_normals, unsigned int* num_texcoords)
{

    tinyobj_material_t* materials = NULL; //NOTE: UNUSED
    size_t num_materials;                 //NOTE: UNUSED


    {
        unsigned int flags = TINYOBJ_FLAG_TRIANGULATE;
        int ret = tinyobj_parse_obj(&elem->attrib, &elem->shapes, &elem->num_shapes, &materials,
                                    &num_materials, data, data_len, flags);
        if (ret != TINYOBJ_SUCCESS) {
            printf("Error: Couldn't parse mesh.\n");
            exit(1);
        }
    }

    *num_vertices  += elem->attrib.num_vertices;
    *num_normals   += elem->attrib.num_normals;
    *num_texcoords += elem->attrib.num_texcoords;
    *num_meshes    += elem->num_shapes;
    //tinyobjloader has dumb variable names: attrib.num_faces =  num_vertices+num_faces
    *num_indices   += elem->attrib.num_faces;
}



void load_obj(struct obj_list_elem elem, int* mesh_offset, int* vert_offset, int* nrml_offset,
                      int* texcoord_offset, int* index_offset, scene* out_scene)
{
    for(int i = 0; i < elem.num_shapes; i++)
    {
        tinyobj_shape_t shape = elem.shapes[i];

        //Get mesh and increment offset.
        mesh* m = (out_scene->meshes) + (*mesh_offset)++;

        m->min[0] = m->min[1] = m->min[2] =  FLT_MAX;
        m->max[0] = m->max[1] = m->max[2] = -FLT_MAX;

        memcpy(m->model, elem.model_mat, 4*4*sizeof(float));

        m->index_offset = *index_offset;
        m->num_indices  =  shape.length*3;
        m->material_index    =  elem.mat_index;
        //memcpy(m->model, elem.model_mat, sizeof(float)*16);
        for(int f = 0; f < shape.length; f++)
        {
            //pretty costly error lol NOTE: do something
            if(elem.attrib.face_num_verts[f+shape.face_offset]!=3)
            {
                //This should never get called because the mesh gets triangulated when loaded.
                printf("Error: the obj loader only supports triangulated meshes!\n");
                exit(1);
            }
            for(int i = 0; i < 3; i++)
            {
                tinyobj_vertex_index_t face_index = elem.attrib.faces[(f+shape.face_offset)*3+i];

                vec3 vertex;
                vertex[0] = elem.attrib.vertices[3*face_index.v_idx+0];
                vertex[1] = elem.attrib.vertices[3*face_index.v_idx+1];
                vertex[2] = elem.attrib.vertices[3*face_index.v_idx+2];

                m->min[0] = vertex[0] < m->min[0] ? vertex[0] : m->min[0]; //X min
                m->min[1] = vertex[1] < m->min[1] ? vertex[1] : m->min[1]; //Y min
                m->min[2] = vertex[2] < m->min[2] ? vertex[2] : m->min[2]; //Z min

                m->max[0] = vertex[0] > m->max[0] ? vertex[0] : m->max[0]; //X max
                m->max[1] = vertex[1] > m->max[1] ? vertex[1] : m->max[1]; //Y max
                m->max[2] = vertex[2] > m->max[2] ? vertex[2] : m->max[2]; //Z max

                ivec3 index;
                index[0] = (*vert_offset)+face_index.v_idx;
                index[1] = (*nrml_offset)+face_index.vn_idx;
                index[2] = (*texcoord_offset)+face_index.vt_idx;
                out_scene->mesh_indices[(*index_offset)][0] = index[0];
                out_scene->mesh_indices[(*index_offset)][1] = index[1];
                out_scene->mesh_indices[(*index_offset)][2] = index[2];

                //xv3_cpy(out_scene->mesh_indices + (*index_offset), index);
                (*index_offset)++;
            }
        }
    }

    //GPU MEMORY ALIGNMENT FUN
    //NOTE: this is done because the gpu stores all vec3s 4 floats for memory alignment
    //      and it is actually faster if they are aligned like this even
    //      though it wastes more memory.
    for(int i = 0; i < elem.attrib.num_vertices; i++)
    {

        memcpy(out_scene->mesh_verts + (*vert_offset),
               elem.attrib.vertices+3*i,
               sizeof(vec3));
        (*vert_offset) += 1;
    }
    for(int i = 0; i < elem.attrib.num_normals; i++)
    {
        memcpy(out_scene->mesh_nrmls + (*nrml_offset),
               elem.attrib.normals+3*i,
               sizeof(vec3));
        (*nrml_offset) += 1;
    }
    //NOTE: the texcoords are already aligned because they only have 2 elements.
    memcpy(out_scene->mesh_texcoords + (*texcoord_offset), elem.attrib.texcoords,
           elem.attrib.num_texcoords*sizeof(vec2));
    (*texcoord_offset) += elem.attrib.num_texcoords;
}

scene* load_scene_json(char* json)
{
    printf("Beginning scene loading...\n");
    scene* out_scene = (scene*) malloc(sizeof(scene));
	JSON_Value *root_value;
    JSON_Object *root_object;
	root_value = json_parse_string(json);
    root_object = json_value_get_object(root_value);


    //Name
    {
        char* name = json_object_get_string(root_object, "name");
        printf("Scene name: %s\n", name);
    }

    //Version
    {//TODO: do something with this.
        int major  = (int)json_object_dotget_number(root_object, "version.major");
        int minor  = (int)json_object_dotget_number(root_object, "version.major");
        char* type =      json_object_dotget_string(root_object, "version.type");
    }

    //Materials
    {
        JSON_Array* material_array = json_object_get_array(root_object, "materials");
        out_scene->num_materials   = json_array_get_count(material_array);
        out_scene->materials = (material*) malloc(out_scene->num_materials*sizeof(material));
        assert(out_scene->num_materials>0);
        for(int i = 0; i < out_scene->num_materials; i++)
        {
            JSON_Object* mat = json_array_get_object(material_array, i);
            xv_x(out_scene->materials[i].colour) = json_object_get_number(mat, "r");
            xv_y(out_scene->materials[i].colour) = json_object_get_number(mat, "g");
            xv_z(out_scene->materials[i].colour) = json_object_get_number(mat, "b");
            out_scene->materials[i].reflectivity = json_object_get_number(mat, "reflectivity");
        }
        printf("Materials: %d\n", out_scene->num_materials);
    }

    //Primitives
    {

        JSON_Object* primitive_object = json_object_get_object(root_object, "primitives");

        //Spheres
        {
            JSON_Array* sphere_array = json_object_get_array(primitive_object, "spheres");
            int num_spheres = json_array_get_count(sphere_array);

            out_scene->spheres = malloc(sizeof(sphere)*num_spheres);
            out_scene->num_spheres = num_spheres;

            for(int i = 0; i < num_spheres; i++)
            {
                JSON_Object* sphere = json_array_get_object(sphere_array, i);
                out_scene->spheres[i].pos[0] = json_object_get_number(sphere, "x");
                out_scene->spheres[i].pos[1] = json_object_get_number(sphere, "y");
                out_scene->spheres[i].pos[2] = json_object_get_number(sphere, "z");
                out_scene->spheres[i].radius = json_object_get_number(sphere, "radius");
                out_scene->spheres[i].material_index = json_object_get_number(sphere, "mat_index");
            }
            printf("Spheres: %d\n", out_scene->num_spheres);
        }

        //Planes
        {
            JSON_Array* plane_array = json_object_get_array(primitive_object, "planes");
            int num_planes = json_array_get_count(plane_array);

            out_scene->planes = malloc(sizeof(plane)*num_planes);
            out_scene->num_planes = num_planes;

            for(int i = 0; i < num_planes; i++)
            {
                JSON_Object* plane = json_array_get_object(plane_array, i);
                out_scene->planes[i].pos[0] = json_object_get_number(plane, "x");
                out_scene->planes[i].pos[1] = json_object_get_number(plane, "y");
                out_scene->planes[i].pos[2] = json_object_get_number(plane, "z");
                out_scene->planes[i].norm[0] = json_object_get_number(plane, "nx");
                out_scene->planes[i].norm[1] = json_object_get_number(plane, "ny");
                out_scene->planes[i].norm[2] = json_object_get_number(plane, "nz");

                out_scene->planes[i].material_index = json_object_get_number(plane, "mat_index");
            }
            printf("Planes: %d\n", out_scene->num_planes);
        }

    }


    //Meshes
    {
        JSON_Array* mesh_array = json_object_get_array(root_object, "meshes");

        int num_meshes = json_array_get_count(mesh_array);

        out_scene->num_meshes = 0;
        out_scene->num_mesh_verts = 0;
        out_scene->num_mesh_nrmls = 0;
        out_scene->num_mesh_texcoords = 0;
        out_scene->num_mesh_indices = 0;


        struct obj_list_elem* first = (struct obj_list_elem*) malloc(sizeof(struct obj_list_elem));
        struct obj_list_elem* current = first;

        //Pre evaluation
        for(int i = 0; i < num_meshes; i++)
        {
            JSON_Object* mesh = json_array_get_object(mesh_array, i);
            char* url = json_object_get_string(mesh, "url");
            long length;
            char* data = load_file(url, &length);
            obj_pre_load(data, length, current, &out_scene->num_meshes, &out_scene->num_mesh_indices,
                         &out_scene->num_mesh_verts, &out_scene->num_mesh_nrmls,
                         &out_scene->num_mesh_texcoords);
            current->mat_index = (int) json_object_get_number(mesh, "mat_index");
            //mat4 model_mat;
            {
                //xm4_identity(model_mat);
                mat4 translation_mat;
                xm4_translatev(translation_mat,
                               json_object_get_number(mesh, "px"),
                               json_object_get_number(mesh, "py"),
                               json_object_get_number(mesh, "pz"));
                mat4 scale_mat;
                xm4_scalev(scale_mat,
                           json_object_get_number(mesh, "sx"),
                           json_object_get_number(mesh, "sy"),
                           json_object_get_number(mesh, "sz"));
                //TODO: add rotation.
                xm4_mul(current->model_mat, translation_mat, scale_mat);
            }
            free(data);

            if(i!=num_meshes-1) //messy but it works
            {
                current->next = (struct obj_list_elem*) malloc(sizeof(struct obj_list_elem));
                current = current->next;
            }
            current->next = NULL;
        }

        //Allocation
        out_scene->meshes          = (mesh*) malloc(sizeof(mesh)*out_scene->num_meshes);
        out_scene->mesh_verts      = (vec3*) malloc(sizeof(vec3)*out_scene->num_mesh_verts);
        out_scene->mesh_nrmls      = (vec3*) malloc(sizeof(vec3)*out_scene->num_mesh_nrmls);
        out_scene->mesh_texcoords  = (vec2*) malloc(sizeof(vec2)*out_scene->num_mesh_texcoords);
        out_scene->mesh_indices    = (ivec3*) malloc(sizeof(ivec3)*out_scene->num_mesh_indices);

        assert(out_scene->meshes!=NULL);
        assert(out_scene->mesh_verts!=NULL);
        assert(out_scene->mesh_nrmls!=NULL);
        assert(out_scene->mesh_texcoords!=NULL);
        assert(out_scene->mesh_indices!=NULL);

        //Parsing and Assignment
        int mesh_offset = 0;
        int vert_offset = 0;
        int nrml_offset = 0;
        int texcoord_offset = 0;
        int index_offset = 0;


        current = first;
        while(current != NULL)
        {

            load_obj(*current, &mesh_offset, &vert_offset, &nrml_offset, &texcoord_offset,
                     &index_offset, out_scene);

            current = current->next;
        }
        printf("%i and %i\n", vert_offset, out_scene->num_mesh_verts);
        assert(mesh_offset==out_scene->num_meshes);
        assert(vert_offset==out_scene->num_mesh_verts);
        assert(nrml_offset==out_scene->num_mesh_nrmls);
        assert(texcoord_offset==out_scene->num_mesh_texcoords);

        assert(index_offset==out_scene->num_mesh_indices);

        printf("Meshes: %d\nVertices: %d\nIndices: %d\n",
               out_scene->num_meshes, out_scene->num_mesh_verts, out_scene->num_mesh_indices);

    }

    out_scene->materials_changed = true;
    out_scene->spheres_changed = true;
    out_scene->planes_changed = true;
    out_scene->meshes_changed = true;


    printf("Finshed scene loading.\n\n");

	json_value_free(root_value);
	return out_scene;
}


scene* load_scene_json_url(char* url)
{
    long variable_doesnt_matter;

    return load_scene_json( load_file(url, &variable_doesnt_matter) ); //TODO: put data
}
