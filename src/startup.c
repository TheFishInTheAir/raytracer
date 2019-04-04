#include <os_abs.h>
#include <stdint.h>
#include <startup.h>
#include <stdio.h>
#include <raytracer.h>
#include <mongoose.h>

#include <ui.h>
#include <ss_raytracer.h>
#include <path_raytracer.h>
#include <spath_raytracer.h>

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
    //TEST THING

    //web_server_test();

    //NORMAL EVERYTHING


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


        os_start_thread(abst, web_server_start, rctx);

#ifdef DEV_MODE
        rctx->event_stack[rctx->event_position++] = SPLIT_PATH_RAYTRACER;
#endif


        //TODO: move
        scene* rscene = load_scene_json_url("scenes/path_obj3.rsc");

        rctx->stat_scene = rscene;
        rctx->num_samples = 128; //NOTE: add input option for this

        ss_raytracer_context* ssrctx = NULL;
        path_raytracer_context* prctx = NULL;
        spath_raytracer_context* sprctx = NULL;
        int current_renderer = -1;
        bool global_up_to_date = false;
        while(should_run)
        {
            if(rctx->event_position)
            {
                if(!global_up_to_date)
                {
                    raytracer_build(rctx); //TODO: cleanup previous stuff.
                    xm4_identity(rctx->stat_scene->camera_world_matrix);//TODO: do something better
                    global_up_to_date = true;
                }
                switch(rctx->event_stack[--rctx->event_position])
                {
                case(SS_RAYTRACER): //TODO: create defines for these
                {
                    printf("Switching To SS Raytracer\n");

                    if(current_renderer==SS_RAYTRACER)
                        break;
                    current_renderer = SS_RAYTRACER;

                    os_draw_weird(abst);
                    os_update(abst);

                    if(ssrctx==NULL)
                        ssrctx = init_ss_raytracer_context(rctx);

                    //if(!ssrctx->up_to_date)
                    //{
                    //ssrctx->up_to_date = true;
                    ss_raytracer_prepass(ssrctx);
                    //}

                    break;
                }
                case(PATH_RAYTRACER):
                {
                    printf("Switching To Path Tracer\n");
                    if(current_renderer==PATH_RAYTRACER)
                        break;
                    current_renderer = PATH_RAYTRACER;

                    os_draw_weird(abst);
                    os_update(abst);

                    if(prctx==NULL)
                        prctx = init_path_raytracer_context(rctx);

                    //if(!ssrctx->up_to_date)
                    //{
                    //ssrctx->up_to_date = true;
                    path_raytracer_prepass(prctx);
                    //}

                    break;
                }
                case(SPLIT_PATH_RAYTRACER):
                {
                    printf("Switching To Split Path Tracer\n");
                    if(current_renderer==SPLIT_PATH_RAYTRACER)
                        break;
                    current_renderer = SPLIT_PATH_RAYTRACER;

                    os_draw_weird(abst);
                    os_update(abst);

                    if(sprctx==NULL)
                        sprctx = init_spath_raytracer_context(rctx);

                    //if(!ssrctx->up_to_date)
                    //{
                    //ssrctx->up_to_date = true;
                    spath_raytracer_prepass(sprctx);
                    //}

                    break;
                }
                }
            }

            switch(current_renderer)
            {
            case(SS_RAYTRACER):
            {
                ss_raytracer_render(ssrctx);
				break;
            }
            case(PATH_RAYTRACER):
            {
                path_raytracer_render(prctx);
				break;
            }
            case(SPLIT_PATH_RAYTRACER):
            {
                spath_raytracer_render(sprctx);
				break;
            }
            }
            os_update(abst);
        }

        //all below shouldn't be a thing.

        raytracer_build(rctx);
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

            //raytracer_refined_render(rctx);
            raytracer_render(rctx);
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
