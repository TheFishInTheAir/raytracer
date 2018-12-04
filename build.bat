::cmd "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\vsdevcmd\ext\vcvars.bat"
echo "NOTE: Must be run with VCVars"
IF NOT EXIST .\build mkdir .\build
pushd .\build

xxd -i ..\src\kernels\test.cl | sed 's/\([0-9a-f]\)$/\0, 0x00/' > .\test.cl.h

xcopy /s ..\res .\res /Y


:: Implement and get working later (Removes CRT overhead)
::cl -Zi -nologo -Gm- -GR- -EHa- -Oi -GS- -Gs9999999 /I..\include /I..\src /I..\libs ..\src\_compiler_sources.c /link /NODEFAULTLIB /SUBSYSTEM:windows -stack:0x100000,0x100000 user32.lib gdi32.lib winmm.lib kernel32.lib
::Zi is for debug.
cl /Zi /Ox /I..\include /I. /I..\src /I..\libs ..\src\_compiler_sources.c /link /SUBSYSTEM:console /LIBPATH:"C:\Program Files (x86)\Intel\OpenCL SDK\6.3\lib\x64" /LIBPATH:..\libs user32.lib gdi32.lib winmm.lib kernel32.lib opencl.lib

popd
