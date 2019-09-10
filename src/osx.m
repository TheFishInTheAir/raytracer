#import <Cocoa/Cocoa.h>
#include <osx.h>
#include <startup.h>
#include <sys/types.h>

#include <os_abs.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>

#if 1
int main()
{
    startup();
}
#endif

typedef struct
{
    unsigned char* bitmap_memory;

    unsigned int width;
    unsigned int height;

    dispatch_queue_t main_queue;

    NSBitmapImageRep* bitmap;
} osx_ctx;
//NOTE: probably not good
static osx_ctx* ctx;


void osx_sleep(int miliseconds)
{
    struct timespec ts;
    ts.tv_sec = miliseconds/1000;
    ts.tv_nsec = (miliseconds%1000)*1000000;
    nanosleep(&ts, NULL);
}

void* osx_get_bitmap_memory()
{
    return ctx->bitmap_memory;
}

int osx_get_time_mili()
{
    int err = 0;
    struct timespec ts;
    if((err = clock_gettime(CLOCK_REALTIME, &ts)))
    {
        printf("ERROR: failed to retrieve time. (osx abstraction) %i", err);
        exit(1);
    }
    return (ts.tv_sec*1000)+(ts.tv_nsec/1000000);
}

int osx_get_width()
{
    return ctx->width;
}
int osx_get_height()
{
    return ctx->height;
}

void initBitmapData(unsigned char* bmap, float offset, unsigned int width, unsigned int height)
{
    int pitch = width*4;
    uint8_t* row = bmap;

    for(int y = 0; y < height; y++)
    {
        uint8_t* pixel = (uint8_t*)row;
        for(int x = 0; x < width; x++)
        {
            *pixel = sin(((float)x+offset)/150)*255;
            ++pixel;

            *pixel = cos(((float)x-offset)/10)*100;
            ++pixel;

            *pixel = cos(((float)y*(offset+1))/50)*255;
            ++pixel;

            *pixel = 255;
            ++pixel;
        }
        row += pitch;
    }
}

void weird_draw()
{
    //printf("I hope this works.\n");
    initBitmapData(ctx->bitmap_memory, 0, ctx->width, ctx->height);
}


//Create OS Virtual Function Struct
os_abs init_osx_abs()
{
    os_abs abstraction;
    abstraction.start_func = &osx_start;
    abstraction.loop_start_func = &osx_loop_start;
    abstraction.update_func = &osx_enqueue_update;
    abstraction.sleep_func = &osx_sleep;
    abstraction.get_bitmap_memory_func = &osx_get_bitmap_memory;
    abstraction.get_time_mili_func = &osx_get_time_mili;
    abstraction.get_width_func = &osx_get_width;
    abstraction.get_height_func = &osx_get_height;
    abstraction.start_thread_func = &osx_start_thread;
    abstraction.draw_weird = &weird_draw;
    return abstraction;
}


@interface CustomView : NSView
@end
@implementation CustomView
- (void)drawRect:(NSRect)dirtyRect {
    CGContextRef gctx = [[NSGraphicsContext currentContext] CGContext];
    CGRect myBoundingBox;
    myBoundingBox = CGRectMake(0,0, ctx->width, ctx->height);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB);
    int bitmapBytesPerRow = ctx->width*4;
    static float thingy = 0;
    //NOTE: not sure if _backBuffer should be stored?? probably not right.
    CGContextRef _backBuffer = CGBitmapContextCreate(ctx->bitmap_memory, ctx->width, ctx->height, 8,
                                                     bitmapBytesPerRow, colorSpace, kCGImageAlphaPremultipliedLast); //NOTE: nonpremultiplied alpha


    CGImageRef backImage = CGBitmapContextCreateImage(_backBuffer);

    CGColorSpaceRelease(colorSpace);

    CGContextDrawImage(gctx, myBoundingBox, backImage);


    CGContextRelease(_backBuffer);
    CGImageRelease(backImage);
}
@end

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end
@implementation AppDelegate

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender
{
    return NSTerminateNow;

}

- (void)applicationDidFinishLaunching:(NSNotification *)notification
{
    id menubar = [[NSMenu new] autorelease];
    id appMenuItem = [[NSMenuItem new] autorelease];
    [menubar addItem:appMenuItem];
    [NSApp setMainMenu:menubar];
    id appMenu = [[NSMenu new] autorelease];
    id appName = [[NSProcessInfo processInfo] processName];
    id quitTitle = [@"Quit " stringByAppendingString:appName];
    id quitMenuItem = [[[NSMenuItem alloc] initWithTitle:quitTitle
                                                  action:@selector(terminate:) keyEquivalent:@"q"] autorelease];
    [appMenu addItem:quitMenuItem];
    [appMenuItem setSubmenu:appMenu];
    NSRect frame = NSMakeRect(0, 0, ctx->width, ctx->height);
    NSUInteger windowStyle = NSWindowStyleMaskTitled;//NSWindowStyleMaskBorderless;
    NSWindow* window  = [[[NSWindow alloc]
                             initWithContentRect:frame
                                       styleMask:windowStyle
                                         backing:NSBackingStoreBuffered
                                           defer:NO] autorelease];

    [window setBackgroundColor:[NSColor grayColor]];
    [window makeKeyAndOrderFront:nil];
    [window cascadeTopLeftFromPoint:NSMakePoint(20,20)];

    CustomView* cv = [[CustomView alloc] initWithFrame:frame];

    [window setContentView:cv];

    initBitmapData(ctx->bitmap_memory, 0, ctx->width, ctx->height);

}
@end

void osx_start()
{
    printf("Initialising OSX context.\n");
    ctx = (osx_ctx*) malloc(sizeof(osx_ctx));

    ctx->width  = 800;
    ctx->height = 800;
    ctx->main_queue = dispatch_get_main_queue();
    ctx->bitmap_memory = malloc(ctx->width*ctx->height*sizeof(int));

    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    NSApp.delegate = [AppDelegate alloc];
}

void osx_loop_start()
{
    printf("Starting OSX Run loop.\n");

    [NSApp activateIgnoringOtherApps:YES];
    [NSApp run];
}

void osx_start_thread(void (*func)(void*), void* data)
{
    pthread_t thread;
    pthread_create(&thread, NULL, (void *(*)(void*))func, data);
}
float offset;
void osx_enqueue_update()
{

    dispatch_async(ctx->main_queue,
                   ^{
                       NSApp.windows[0].title =
                           [NSString stringWithFormat:@"Pathtracer %f", offset];
                       CustomView* view = (CustomView*) NSApp.windows[0].contentView;
                           [view setNeedsDisplay:YES];

                       [NSApp.windows[0] display]; //This should also call display on view
                   });
}

void _test_thing(void* data) //TODO: CLEANUP
{
    offset = 40.0f;
    printf("test start\n");
    while(true)
    {
        osx_sleep(1);
        initBitmapData(ctx->bitmap_memory, offset, ctx->width, ctx->height);
        osx_enqueue_update();
        offset += 10.0f;
        if(offset>300)
            offset = 0;
        printf("test loop\n");
    }
}

#if 0
int main ()
{
    osx_start();

    osx_loop_start();

    return 0;
}
#endif
