#include <os_abs.h>
#include <stdint.h>
#include <startup.h>
#include <stdio.h>
#include <raytracer.h>




#ifdef WIN32
#include <win32.h>
#else
#include <osx.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#endif

//#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <geom.h>
#include <parallel.h>
#include <loader.h>
#define NUM_SPHERES 5
#define NUM_PLANES  1

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

    const int pitch = width*4;

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

#ifndef _WIN32
char kbhit()
{
    static char initialised = false;
    //NOTE: we are never going to need to actually echo the characters
    if(!initialised)
    {
        initialised = true;
        struct termios term, old;
        tcgetattr(STDIN_FILENO, &old);
        term = old;
        term.c_lflag &= -(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &term);
    }
    struct timeval tv;
    fd_set rdfs;

    tv.tv_sec = 0;
    tv.tv_usec = 0;

    FD_ZERO(&rdfs);
    FD_SET(STDIN_FILENO, &rdfs);

    select(STDIN_FILENO+1, &rdfs, NULL, NULL, &tv);
    return FD_ISSET(STDIN_FILENO, &rdfs);
}
#endif


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

    char isMeme = 'y';
    //scanf("%c", &isMeme);

    if(isMeme=='y')
    {
        const int width = os_get_width(abst);
        const int height = os_get_height(abst);

        const int pitch = width *4;
        uint32_t* row = (uint32_t*)os_get_bitmap_memory(abst);

        cl_info();

        rcl_ctx* rcl = (rcl_ctx*) malloc(sizeof(rcl_ctx));
        create_context(rcl);

        raytracer_context* rctx = raytracer_init((unsigned int)width, (unsigned int)height,
                                                 row, rcl);
		//scene* rscene = (scene*) malloc(sizeof(scene));
        scene* rscene = load_scene_json_url("scenes/path_test_2.rsc");

        rctx->stat_scene = rscene;
        rctx->num_samples = 64; //NOTE: add input option for this

        raytracer_prepass(rctx);

        xm4_identity(rctx->stat_scene->camera_world_matrix);

        float dist = 0.f;


        int _timer_store = 0;
        int _timer_counter = 0;
        float _timer_average = 0.0f;
        printf("Rendering:\n\n");

        /* static float t = 0.0f; */
        /* t += 0.0005f; */
        /* dist = sin(t)+1; */
        /* //mat4 temp; */
        /* xm4_translatev(rctx->stat_scene->camera_world_matrix, 0, dist, 0); */
        int real_start = os_get_time_mili(abst);
        while(should_run)
        {

            if(should_pause)
                continue;
            int last_time = os_get_time_mili(abst);

            if(kbhit())
            {
                switch (getc(stdin))
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

            raytracer_refined_render(rctx);
            if(rctx->render_complete)
            {
                printf("\n\nRender took: %02i ms (%d samples)\n\n",
                       os_get_time_mili(abst)-real_start, rctx->num_samples);
                break;
            }


            int mili = os_get_time_mili(abst)-last_time;
            _timer_store += mili;
            _timer_counter++;
            printf("\rFrame took: %02i ms, average per 20 frames: %0.2f, avg fps: %03.2f (%d/%d)    ",
                   mili, _timer_average, 1000.0f/_timer_average,
                   rctx->current_sample, rctx->num_samples);
            fflush(stdout);
            if(_timer_counter>20)
            {
                _timer_counter = 0;
                _timer_average = (float)(_timer_store)/20.f;
                _timer_store = 0;
            }
            os_update(abst);
        }

    }


}

int startup() //main function called from win32 abstraction
{
#ifdef WIN32
    abst = init_win32_abs();
#else
    abst = init_osx_abs();
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
