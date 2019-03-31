::cmd "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\vsdevcmd\ext\vcvars.bat"
echo "NOTE: Must be run with VCVars"
IF NOT EXIST .\build mkdir .\build
pushd .\build
::
cat  "..\src\kernels\util.cl" "..\src\kernels\collision.cl"  "..\src\kernels\irradiance_cache.cl" "..\src\kernels\ray.cl" "..\src\kernels\path.cl" "..\src\kernels\kdtree.cl" > all_kernels.cl
xxd -i all_kernels.cl | sed 's/\([0-9a-f]\)$/\0, 0x00/' > .\test.cl.h

xxd -i ..\src\ui\index.html | sed 's/\([0-9a-f]\)$/\0, 0x00/' > .\index.html.h
xxd -i ..\src\ui\style.css | sed 's/\([0-9a-f]\)$/\0, 0x00/' > .\style.css.h
xxd -i ..\src\ui\ocp_li.woff | sed 's/\([0-9a-f]\)$/\0, 0x00/' > .\ocp_li.woff.h
cat  ".\index.html.h" ".\ocp_li.woff.h" ".\style.css.h" > ui_web.h

xcopy /s ..\res .\res\ /Y


:: Implement and get working later (Removes CRT overhead)
::cl -Zi -nologo -Gm- -GR- -EHa- -Oi -GS- -Gs9999999 /I..\include /I..\src /I..\libs ..\src\_compiler_sources.c /link /NODEFAULTLIB /SUBSYSTEM:windows -stack:0x100000,0x100000 user32.lib gdi32.lib winmm.lib kernel32.lib
::Zi is for debug.
::NOTE: removed /Ox
cl /Zi /Ox /I..\include /I. /I..\src /I..\libs ..\src\_compiler_sources.c /link /SUBSYSTEM:console /LIBPATH:"C:\Program Files (x86)\Intel\OpenCL SDK\6.3\lib\x64" /LIBPATH:..\libs user32.lib gdi32.lib winmm.lib kernel32.lib opencl.lib advapi32.lib

popd
