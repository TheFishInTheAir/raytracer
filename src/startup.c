#include <os_abs.h>
#include <stdint.h>
#include <startup.h>
#include <stdio.h>
#include <raytracer.h>
#define WIN32 //BECAUSE CL IS FUCKING GAY

#ifdef WIN32
#include <win32.h>
#endif

//#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <geom.h>
#include <parallel.h>

#define NUM_SPHERES 5
#define STRFY(x) #x
#define DBL_STRFY(x) STRFY(x)

os_abs abst;

void cast_rays(int width, int height, uint32_t* bmap)
{


    // unsigned width = 640, height = 480;
    // Vec3f *image = new Vec3f[width * height], *pixel = image;
    //float invWidth = 1 / (float)width, invHeight = 1 / (float)height;
    //float fov = 30, aspectratio = width / (float)height;
    //float angle = tan(M_PI * 0.5 * fov / 180.);


    static float dist = 5.0f;

    sphere s;
    xv_x(s.pos) = 0.0f;
    xv_y(s.pos) = 0.0f;
    xv_z(s.pos) = -dist;
    s.radius = 1.0f;

    if(dist<2.0f)
        dist = 10.0f;
    dist -= 0.05f;


    int last_time = os_get_time_mili(abst);

    const pitch = width*4;

    int y = 0;
    int x = 0;
    for(y = 0; y < height; y++)
    {
        uint32_t* pixel = (uint32_t*)bmap;
        for(x = 0; x < width; x++)
        {
            ray out_ray = generate_ray(x, y, width, height, 90);
            float dist =  does_collide_sphere(s, out_ray);
            *pixel = dist != -1.0f ? 0x00ffffff & (int) dist*100 : 0x00000000;
            //*pixel = 0x000000ff | ((uint32_t)((uint8_t)(y)))<<16;
            pixel++;
        }
        bmap += width;
    }
    /* float stest = 0.0f; */

    /* // compute 1e8 times either Sqrt(x) or its emulation as Pow(x, 0.5) */
    /* for (float d = 0; d < width*height*2; d += 1) */
    /*     // s += Math.Sqrt(d);  // <- uncomment it to test Sqrt */
    /*     stest += sqrt(d*d); // <- uncomment it to test Pow */


    printf("frame took: %i ms\n", os_get_time_mili(abst)-last_time);

}

bool should_run = true;
bool should_pause = false;
void loop_exit()
{
    should_run = false;
}
void loop_pause()
{
    should_pause = !should_pause;
}

void run(void* unnused_rn)
{
	OutputDebugStringA("Starded Render Thread\n");

	OutputDebugStringA("Meme? (y/n)\n");

    char isMeme = 'y';
    //scanf("%c", &isMeme);

    if(isMeme=='y')
    {
		OutputDebugStringA("YALL ARE MEMERS\n");
        const int width = os_get_width(abst);
        const int height = os_get_height(abst);

        const int pitch = width *4;
        uint32_t* row = (uint32_t*)os_get_bitmap_memory(abst);

        /* while(1) */
        /* { */
        /*     row = (uint32_t*)win32_get_bitmap_memory(); */
        /*     cast_rays(width, height, row); */

        /*     win32_update(); */

        /* } */
        cl_info();

        rcl_ctx* rcl = (rcl_ctx*) malloc(sizeof(rcl_ctx));
        create_context(rcl);
        /*char* kernels[] = {"magenta_test"};
        char* macros[]  = {"#define SCENE_NUM_SPHERES " DBL_STRFY(NUM_SPHERES)};
        rcl_program program;
        load_program_url(&ctx,
                         "C:\\Users\\Ethan Breit\\AppData\\Roaming\\Emacs\\Western\\10\\Science\\Raytracer\\src\\kernels\\test.cl",
                         kernels, 1, &program,
                         macros, 1);*/
        raytracer_context* rctx = raytracer_init((unsigned int)width, (unsigned int)height,
                                                 row, rcl);
        scene* rscene = (scene*) malloc(sizeof(scene));
        rscene->num_spheres = 5;
        rscene->spheres = (sphere*) malloc(sizeof(sphere)*5);
        rctx->stat_scene = rscene;

        raytracer_prepass(rctx);
        while(should_run)
        {
            if(should_pause)
                continue;

            if(kbhit())
            {
                switch (getch())
                {
                case 'c':
                    exit(1);
                    break;
                case 27: //ESCAPE
                    exit(1);
                    break;
                default:
                    break;
                }
            }

            static float dist = -5.0f;
            static bool state = false;
            sphere s;
            xv_x(s.pos) = 0.0f;
            xv_y(s.pos) = 0.0f;
            xv_z(s.pos) = dist;
            s.radius = 1.0f;
            sphere s2;
            xv_x(s2.pos) = (dist*2)+10;
            xv_y(s2.pos) = 0.0f;
            xv_z(s2.pos) = -10.0f;
            s2.radius = 0.6f;
            sphere s3;
            xv_x(s3.pos) = sin(dist/7)*8;
            xv_y(s3.pos) = cos(dist/7)*8;
            xv_z(s3.pos) = -11.0f;
            s3.radius    = 0.5f;
            sphere s4;
            xv_x(s4.pos) = sin(dist/3+1)*3;
            xv_y(s4.pos) = cos(dist/3+1)*3;
            xv_z(s4.pos) = -11.0f;
            s4.radius    = 0.5f;
            sphere s5;
            xv_x(s5.pos) = sin(dist/5+2)*5;
            xv_y(s5.pos) = cos(dist/5+2)*5;
            xv_z(s5.pos) = -11.0f;
            s5.radius    = 0.5f;



            if(state)
            {
                dist += 0.05f;
                if(dist>-2.0f)
                    state = false;
            }
            else
            {
                dist -= 0.05f;
                if(dist<-10.0f)
                    state=true;
            }

            rscene->spheres[0] = s;
            rscene->spheres[1] = s2;
            rscene->spheres[2] = s3;
            rscene->spheres[3] = s4;
            rscene->spheres[4] = s5;


            //NOTE: has test hardcoded url.
            int last_time = os_get_time_mili(abst);

            raytracer_render(rctx);
            /*test_sphere_raytracer(&ctx, &program, spheres, NUM_SPHERES,
              row, width, height);*/
            printf("frame took: %i ms\n", os_get_time_mili(abst)-last_time);

            os_update(abst);
        }

    }


}


int startup() //mainfunction called from win32 abstraction
{
#ifdef WIN32
    abst = init_win32_abs();
#endif
    os_start(abst);
    os_start_thread(abst, run, NULL);
    //win32_start_thread(run, NULL);

    os_loop_start(abst);
    return 0;
    /*
    printf("Hello World\n");
    testWin32();
    return 0;*/
}
