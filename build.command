echo cool fancy osx build script
#clang ./src/_compiler_sources.c -I./libs/ -I./src/ -I./include/
#gcc ./src/osx.c -fobjc-arc -framework Cocoa -x objective-c -o test
if ! -a ./build ; then
   mkdir build
fi

pushd ./build ;

cat  "../src/kernels/util.cl" "../src/kernels/collision.cl"  "../src/kernels/irradiance_cache.cl" "../src/kernels/ray.cl" "../src/kernels/path.cl" > all_kernels.cl
xxd -i all_kernels.cl | sed 's/\([0-9a-f]\)$/\0, 0x00/' > ./test.cl.h
cp -R ../res ./

clang -c -framework Cocoa ../src/osx.m -I../include -o osx.o
clang -c ../src/_compiler_sources.c -I./ -I../libs -I../include -I../src -o ./_compiler_sources.o
#ld  osx.obj _compiler_sources.obj -framework Cocoa -framework Foundation -framework OpenCL -execute -lc -L/usr/local/lib -lSystem -lto_library /Library/Developer/CommandLineTools/usr/lib/libLTO.dylib -o _good
ld -arch x86_64 -macosx_version_min 10.14.0 -o _compiler_sources -framework Cocoa -framework OpenCL osx.o _compiler_sources.o -lSystem -lc

#clang -v -framework Cocoa ../src/osx.m -I../include -o osx.exe

popd

#./build/_compiler_sources
