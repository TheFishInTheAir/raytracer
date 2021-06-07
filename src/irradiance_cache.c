/******************************************/
/* NOTE: Irradiance Caching is Incomplete */
/******************************************/

#include <irradiance_cache.h>
#include <raytracer.h>
#include <parallel.h>


void ic_init(raytracer_context* rctx)
{
    rctx->ic_ctx->cl_standard_format.image_channel_order     = CL_RGBA;
    rctx->ic_ctx->cl_standard_format.image_channel_data_type = CL_FLOAT;
    
    rctx->ic_ctx->cl_standard_descriptor.image_type = CL_MEM_OBJECT_IMAGE2D;
    rctx->ic_ctx->cl_standard_descriptor.image_width = rctx->width;
    rctx->ic_ctx->cl_standard_descriptor.image_height = rctx->height;
    rctx->ic_ctx->cl_standard_descriptor.image_depth  = 0;
    rctx->ic_ctx->cl_standard_descriptor.image_array_size  = 0;
    rctx->ic_ctx->cl_standard_descriptor.image_row_pitch  = 0;
    rctx->ic_ctx->cl_standard_descriptor.num_mip_levels = 0;
    rctx->ic_ctx->cl_standard_descriptor.num_samples = 0;
    rctx->ic_ctx->cl_standard_descriptor.buffer = NULL;
    
    rctx->ic_ctx->octree.node_count = 1; //root
    //TODO: add as parameter
    rctx->ic_ctx->octree.max_depth = 8;  //arbitrary
    rctx->ic_ctx->octree.width     = 15; //arbitrary
    
    rctx->ic_ctx->octree.root = (ic_octree_node*) malloc(sizeof(ic_octree_node));
    rctx->ic_ctx->octree.root->min[0] = (float)-rctx->ic_ctx->octree.width;
    rctx->ic_ctx->octree.root->min[1] = (float)-rctx->ic_ctx->octree.width;
    rctx->ic_ctx->octree.root->min[2] = (float)-rctx->ic_ctx->octree.width;
    rctx->ic_ctx->octree.root->max[0] = (float) rctx->ic_ctx->octree.width;
    rctx->ic_ctx->octree.root->max[1] = (float) rctx->ic_ctx->octree.width;
    rctx->ic_ctx->octree.root->max[2] = (float) rctx->ic_ctx->octree.width;
    rctx->ic_ctx->octree.root->leaf = false;
    rctx->ic_ctx->octree.root->active = false;
}

void ic_octree_init_leaf(ic_octree_node* node, ic_octree_node* parent, unsigned int i)
{
    float xhalf = (parent->max[0]-parent->min[0])/2;
    float yhalf = (parent->max[1]-parent->min[1])/2;
    float zhalf = (parent->max[2]-parent->min[2])/2;
    node->active = false;
    
    node->leaf = true;
    for(int i = 0; i < 8; i++)
        node->data.branch.children[i] = NULL;
    node->min[0] = parent->min[0] + ( (i&4) ? xhalf : 0);
    node->min[1] = parent->min[1] + ( (i&2) ? yhalf : 0);
    node->min[2] = parent->min[2] + ( (i&1) ? zhalf : 0);
    node->max[0] = parent->max[0] - (!(i&4) ? xhalf : 0);
    node->max[1] = parent->max[1] - (!(i&2) ? yhalf : 0);
    node->max[2] = parent->max[2] - (!(i&1) ? zhalf : 0);
}

void ic_octree_make_branch(ic_octree* tree, ic_octree_node* node)
{
    
    node->leaf = false;
    for(int i = 0; i < 8; i++)
    {
        node->data.branch.children[i] = malloc(sizeof(ic_octree_node));
        ic_octree_init_leaf(node->data.branch.children[i], node, i);
        tree->node_count++;
    }
}

//TODO: test if points are the same
void _ic_octree_rec_resolve(ic_context* ictx, ic_octree_node* leaf, unsigned int node1, unsigned int node2,
                            unsigned int depth)
{
    if(depth > ictx->octree.max_depth)
    {
        //TODO: just group buffers together
        printf("ERROR: octree reached max depth when trying to resolve collision. (INCOMPLETE)\n");
        exit(1);
    }
    vec3 mid_point;
    xv_sub(mid_point, leaf->max, leaf->min, 3);
    xv_divieq(mid_point, 2, 3);
    unsigned int i1 =
        ((mid_point[0]<ictx->ir_buf[node1].point[0])<<2) |
        ((mid_point[1]<ictx->ir_buf[node1].point[1])<<1) |
        ((mid_point[2]<ictx->ir_buf[node1].point[2]));
    unsigned int i2 =
        ((mid_point[0]<ictx->ir_buf[node2].point[0])<<2) |
        ((mid_point[1]<ictx->ir_buf[node2].point[1])<<1) |
        ((mid_point[2]<ictx->ir_buf[node2].point[2]));
    ic_octree_make_branch(&ictx->octree, leaf);
    if(i1==i2)
        _ic_octree_rec_resolve(ictx, leaf->data.branch.children[i1], node1, node2, depth+1);
    else
    { //happiness
        leaf->data.branch.children[i1]->data.leaf.buffer_offset = node1;
        leaf->data.branch.children[i1]->data.leaf.num_elems = 1;
        leaf->data.branch.children[i2]->data.leaf.buffer_offset = node2;
        leaf->data.branch.children[i2]->data.leaf.num_elems = 1;
    }
}

void _ic_octree_rec_insert(ic_context* ictx, ic_octree_node* node, unsigned int v_ptr, unsigned int depth)
{
    if(node->leaf && !node->active)
    {
        node->active = true;
        node->data.leaf.buffer_offset = v_ptr;
        node->data.leaf.num_elems     = 1; //TODO: add suport for more than 1.
        return;
    }
    else if(node->leaf)
    {
        //resolve
        _ic_octree_rec_resolve(ictx, node, v_ptr, node->data.leaf.buffer_offset, depth+1);
    }
    else
    {
        ic_octree_node* new_node = node->data.branch.children[
            ((ictx->ir_buf[node->data.leaf.buffer_offset].point[0]<ictx->ir_buf[v_ptr].point[0])<<2) |
                ((ictx->ir_buf[node->data.leaf.buffer_offset].point[1]<ictx->ir_buf[v_ptr].point[1])<<1) |
                ((ictx->ir_buf[node->data.leaf.buffer_offset].point[2]<ictx->ir_buf[v_ptr].point[2]))];
        _ic_octree_rec_insert(ictx, new_node, v_ptr, depth+1);
    }
}

void ic_octree_insert(ic_context* ictx, vec3 point, vec3 normal)
{
    if(ictx->ir_buf_current_offset==ictx->ir_buf_size) //TODO: dynamically resize or do something else
    {
        printf("ERROR: irradiance buffer is full!\n");
        exit(1);
    }
    ic_ir_value irradiance_value; //TODO: EVALUATE THIS
    irradiance_value.rad = 0.f; //Gets rid of error, this doesn't work anyways so its good enough.
    ictx->ir_buf[ictx->ir_buf_current_offset++] = irradiance_value;
    _ic_octree_rec_insert(ictx, ictx->octree.root, ictx->ir_buf_current_offset, 0);
}

//NOTE: outBuffer is only bools but using char for safety accross compilers.
//      Also assuming that buf is grayscale
void dither(float* buf, const int width, const int height)
{
    for(int y = 0; y < height; y++ )
    {
        for(int x = 0; x < width; x++ )
        {
            float oldpixel  = buf[x+y*width];
            float newpixel  = oldpixel>0.5f ? 1 : 0;
            buf[x+y*width]  = newpixel;
            float err = oldpixel - newpixel;
            
            if( (x != (width-1)) && (x != 0) && (y != (height-1)) )
            {
                buf[(x+1)+(y  )*width] = buf[(x+1)+(y  )*width] + err  * (7.f / 16.f);
                buf[(x-1)+(y+1)*width] = buf[(x-1)+(y+1)*width] + err  * (3.f / 16.f);
                buf[(x  )+(y+1)*width] = buf[(x  )+(y+1)*width] + err  * (5.f / 16.f);
                buf[(x+1)+(y+1)*width] = buf[(x+1)+(y+1)*width] + err  * (1.f / 16.f);
            }
        }
    }
}


void get_geom_maps(raytracer_context* rctx, cl_mem positions, cl_mem normals)
{
    int err;
    
    cl_kernel kernel = rctx->program->raw_kernels[IC_SCREEN_TEX_KRNL_INDX];
    
    float zeroed[] = {0., 0., 0., 1.};
    float* result = matvec_mul(rctx->stat_scene->camera_world_matrix, zeroed);

    clSetKernelArg(kernel, 0, sizeof(cl_mem),  &positions);
    clSetKernelArg(kernel, 1, sizeof(cl_mem),  &normals);
    clSetKernelArg(kernel, 2, sizeof(int),     &rctx->width);
    clSetKernelArg(kernel, 3, sizeof(int),     &rctx->height);
    clSetKernelArg(kernel, 4, sizeof(cl_mem),  &rctx->cl_ray_buffer);
    clSetKernelArg(kernel, 5, sizeof(vec4),    result);
    clSetKernelArg(kernel, 6, sizeof(cl_mem),  &rctx->stat_scene->cl_material_buffer);
    clSetKernelArg(kernel, 7, sizeof(cl_mem),  &rctx->stat_scene->cl_sphere_buffer);
    clSetKernelArg(kernel, 8, sizeof(cl_mem),  &rctx->stat_scene->cl_plane_buffer);
    clSetKernelArg(kernel, 9, sizeof(cl_mem),  &rctx->stat_scene->cl_mesh_buffer);
    clSetKernelArg(kernel, 10, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_index_buffer);
    clSetKernelArg(kernel, 11, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_vert_buffer);
    clSetKernelArg(kernel, 12, sizeof(cl_mem), &rctx->stat_scene->cl_mesh_nrml_buffer);
    
    size_t global = rctx->width*rctx->height;
    size_t local = 0;
    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(local), &local, NULL);
    ASRT_CL("Failed to Retrieve Kernel Work Group Info");
    
    err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1, NULL, &global,
                                 NULL, 0, NULL, NULL);
    ASRT_CL("Failed to Enqueue kernel IC_SCREEN_TEX");
    
    //Wait for completion
    err = clFinish(rctx->rcl->commands);
    ASRT_CL("Something happened while waiting for kernel to finish");
}

void gen_mipmap_chain_gb(raytracer_context* rctx, cl_mem texture,
                         ic_mipmap_gb* mipmaps, int num_mipmaps)
{
    int err;
    unsigned int width  = rctx->width;
    unsigned int height = rctx->height;
    cl_kernel kernel = rctx->program->raw_kernels[IC_MIP_REDUCE_KRNL_INDX];
    for(int i = 0; i < num_mipmaps; i++)
    {
        mipmaps[i].width  = width;
        mipmaps[i].height = height;
        
        if(i==0)
        {
            mipmaps[0].cl_image_ref = texture;
            
            height /= 2;
            width /= 2;
            continue;
        }
        
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &mipmaps[i-1].cl_image_ref);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &mipmaps[i].cl_image_ref);
        clSetKernelArg(kernel, 2, sizeof(int),    &width);
        clSetKernelArg(kernel, 3, sizeof(int),    &height);
        
        size_t global = width*height;
        size_t local = get_workgroup_size(rctx, kernel);
        
        err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1,
                                     NULL, &global, NULL, 0, NULL, NULL);
        ASRT_CL("Failed to Enqueue kernel IC_MIP_REDUCE");
        
        height /= 2;
        width /= 2;
        //Wait for completion before doing next mip
        err = clFinish(rctx->rcl->commands);
        ASRT_CL("Something happened while waiting for kernel to finish");
    }
}

void upsample_mipmaps_f(raytracer_context* rctx, cl_mem texture,
                        ic_mipmap_f* mipmaps, int num_mipmaps)
{
    int err;
    
    cl_mem* full_maps = (cl_mem*) alloca(sizeof(cl_mem)*num_mipmaps);
    for(int i = 1; i < num_mipmaps; i++)
    {
        full_maps[i] = gen_grayscale_buffer(rctx, 0, 0);
    }
    full_maps[0] = texture;
    { //Upsample
        for(int i = 0; i < num_mipmaps; i++) //First one is already at proper resolution
        {
            cl_kernel kernel = rctx->program->raw_kernels[IC_MIP_S_UPSAMPLE_SCALED_KRNL_INDX];
            
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &mipmaps[i].cl_image_ref);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &full_maps[i]); //NOTE: need to generate this for the function
            clSetKernelArg(kernel, 2, sizeof(int),    &i);
            clSetKernelArg(kernel, 3, sizeof(int),    &rctx->width);
            clSetKernelArg(kernel, 4, sizeof(int),    &rctx->height);
            
            size_t global = rctx->width*rctx->height;
            size_t local = get_workgroup_size(rctx, kernel);
            
            err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1,
                                         NULL, &global, NULL, 0, NULL, NULL);
            ASRT_CL("Failed to Enqueue kernel IC_MIP_S_UPSAMPLE_SCALED");
            
        }
        err = clFinish(rctx->rcl->commands);
        ASRT_CL("Something happened while waiting for kernel to finish");
    }
    printf("Upsampled Discontinuity Mipmaps\nAveraging Upsampled Discontinuity Mipmaps\n");
    
    { //Average
        int total = num_mipmaps;
        for(int i = 0; i < num_mipmaps; i++) //First one is already at proper resolution
        {
            cl_kernel kernel = rctx->program->raw_kernels[IC_FLOAT_AVG_KRNL_INDX];
            
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &full_maps[i]);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &texture);
            clSetKernelArg(kernel, 2, sizeof(int),    &rctx->width);
            clSetKernelArg(kernel, 3, sizeof(int),    &rctx->height);
            clSetKernelArg(kernel, 4, sizeof(int),    &total);
            
            size_t global = rctx->width*rctx->height;
            size_t local = 0;
            err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                           sizeof(local), &local, NULL);
            ASRT_CL("Failed to Retrieve Kernel Work Group Info");
            
            err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1,
                                         NULL, &global, NULL, 0, NULL, NULL);
            ASRT_CL("Failed to Enqueue kernel IC_FLOAT_AVG");
            
            err = clFinish(rctx->rcl->commands);
            ASRT_CL("Something happened while waiting for kernel to finish");
        }
    }
    for(int i = 1; i < num_mipmaps; i++)
    {
        err = clReleaseMemObject(full_maps[i]);
        ASRT_CL("Failed to cleanup fullsize mipmaps");
    }
}

void gen_discontinuity_maps(raytracer_context* rctx, ic_mipmap_gb* pos_mipmaps,
                            ic_mipmap_gb* nrm_mipmaps, ic_mipmap_f* disc_mipmaps,
                            int num_mipmaps)
{
    int err;
    //TODO: tune k and intensity
    const float k = 1.6f;
    const float intensity = 0.02f;
    for(int i = 0; i < num_mipmaps; i++)
    {
        cl_kernel kernel = rctx->program->raw_kernels[IC_GEN_DISC_KRNL_INDX];
        disc_mipmaps[i].width  = pos_mipmaps[i].width;
        disc_mipmaps[i].height = pos_mipmaps[i].height;
        
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &pos_mipmaps[i].cl_image_ref);
        
        
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &nrm_mipmaps[i].cl_image_ref);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &disc_mipmaps[i].cl_image_ref);
        clSetKernelArg(kernel, 3, sizeof(float),  &k);
        clSetKernelArg(kernel, 4, sizeof(float),  &intensity);
        clSetKernelArg(kernel, 5, sizeof(int),    &pos_mipmaps[i].width);
        clSetKernelArg(kernel, 6, sizeof(int),    &pos_mipmaps[i].height);
        
        size_t global = pos_mipmaps[i].width*pos_mipmaps[i].height;
        size_t local = get_workgroup_size(rctx, kernel);
        
        err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1,
                                     NULL, &global, NULL, 0, NULL, NULL);
        ASRT_CL("Failed to Enqueue kernel IC_GEN_DISC");
        
    }
    err = clFinish(rctx->rcl->commands);
    ASRT_CL("Something happened while waiting for kernel to finish");
}

void ic_screenspace(raytracer_context* rctx)
{
    int err;
    
    
    vec4*   pos_tex = (vec4*) malloc(rctx->width*rctx->height*sizeof(vec4));
    vec4*   nrm_tex = (vec4*) malloc(rctx->width*rctx->height*sizeof(vec4));
    float*  c_fin_disc_map = (float*) malloc(rctx->width*rctx->height*sizeof(float));
    
    ic_mipmap_gb pos_mipmaps [NUM_MIPMAPS]; //A lot of buffers
    ic_mipmap_gb nrm_mipmaps [NUM_MIPMAPS];
    ic_mipmap_f  disc_mipmaps[NUM_MIPMAPS];
    cl_mem       fin_disc_map;
    //OpenCL
    cl_mem cl_pos_tex;
    cl_mem cl_nrm_tex;
    cl_image_desc cl_mipmap_descriptor = rctx->ic_ctx->cl_standard_descriptor;
    
    { //OpenCL Init
        cl_pos_tex = gen_rgb_image(rctx, 0,0);
        cl_nrm_tex = gen_rgb_image(rctx, 0,0);
        
        fin_disc_map = gen_grayscale_buffer(rctx, 0,0);
        zero_buffer_img(rctx, fin_disc_map, sizeof(float), 0, 0);
        
        
        unsigned int width  = rctx->width,
        height = rctx->height;
        for(int i = 0; i < NUM_MIPMAPS; i++)
        {
            if(i!=0)
            {
                pos_mipmaps[i].cl_image_ref = gen_rgb_image(rctx, width, height);
                nrm_mipmaps[i].cl_image_ref = gen_rgb_image(rctx, width, height);
            }
            disc_mipmaps[i].cl_image_ref = gen_grayscale_buffer(rctx, width, height);
            
            width /= 2;
            height /= 2;
        }
    }
    printf("Initialised Irradiance Cache Screenspace Buffers\nGetting Screenspace Geometry Data\n");
    get_geom_maps(rctx, cl_pos_tex, cl_nrm_tex);
    printf("Got Screenspace Geometry Data\nGenerating MipMaps\n");
    gen_mipmap_chain_gb(rctx, cl_pos_tex,
                        pos_mipmaps, NUM_MIPMAPS);
    gen_mipmap_chain_gb(rctx, cl_nrm_tex,
                        nrm_mipmaps, NUM_MIPMAPS);
    printf("Generated MipMaps\nGenerating Discontinuity Map for each Mip\n");
    gen_discontinuity_maps(rctx, pos_mipmaps, nrm_mipmaps, disc_mipmaps, NUM_MIPMAPS);
    printf("Generated Discontinuity Map for each Mip\nUpsampling Discontinuity Mipmaps\n");
    upsample_mipmaps_f(rctx, fin_disc_map, disc_mipmaps, NUM_MIPMAPS);
    printf("Averaged Upsampled Discontinuity Mipmaps\nRetrieving Discontinuity Data\n");
    retrieve_buf(rctx, fin_disc_map, c_fin_disc_map,
                 rctx->width*rctx->height*sizeof(float));
    retrieve_image(rctx, cl_pos_tex, pos_tex, 0, 0);
    retrieve_image(rctx, cl_pos_tex, pos_tex, 0, 0);
    
    printf("Retrieved Discontinuity Data\nDithering Discontinuity Map\n");
    //NOTE: read buffer is blocking so we don't need clFinish
    dither(c_fin_disc_map, rctx->width, rctx->height);
    err = clEnqueueWriteBuffer(rctx->rcl->commands, fin_disc_map,
                               CL_TRUE, 0,
                               rctx->width*rctx->height*sizeof(float),
                               c_fin_disc_map, 0, 0, NULL);
    ASRT_CL("Failed to write dithered discontinuity map");
    
    
    //INSERT
    cl_kernel kernel = rctx->program->raw_kernels[BLIT_FLOAT_OUTPUT_INDX];
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rctx->cl_output_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &fin_disc_map);
    clSetKernelArg(kernel, 2, sizeof(int),    &rctx->width);
    clSetKernelArg(kernel, 3, sizeof(int),    &rctx->height);
    
    size_t global = rctx->width*rctx->height;
    size_t local = 0;
    err = clGetKernelWorkGroupInfo(kernel, rctx->rcl->device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(local), &local, NULL);
    ASRT_CL("Failed to Retrieve Kernel Work Group Info");
    
    err = clEnqueueNDRangeKernel(rctx->rcl->commands, kernel, 1,
                                 NULL, &global, NULL, 0, NULL, NULL);
    ASRT_CL("Failed to Enqueue kernel BLIT_FLOAT_OUTPUT_INDX");
    
    clFinish(rctx->rcl->commands);
    
    err = clEnqueueReadBuffer(rctx->rcl->commands, rctx->cl_output_buffer, CL_TRUE, 0,
                              rctx->width*rctx->height*sizeof(int), rctx->output_buffer, 0, NULL, NULL );
    ASRT_CL("Failed to Read Output Buffer");
    printf("test!!\n");
    
    
}
