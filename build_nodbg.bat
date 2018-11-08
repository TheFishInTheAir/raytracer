::cmd "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\vsdevcmd\ext\vcvars.bat"
echo "NOTE: Must be run with VCVars"
echo "No Debug Symbols"
IF NOT EXIST .\build mkdir .\build
pushd .\build

:: NOTE: There is actual wizardry happening here I literally copied these from the Handmade hero forums..
:: Implement and get working later (Removes CRT overhead)
::cl -Zi -nologo -Gm- -GR- -EHa- -Oi -GS- -Gs9999999 /I..\include /I..\src /I..\libs ..\src\_compiler_sources.c /link /NODEFAULTLIB /SUBSYSTEM:windows -stack:0x100000,0x100000 user32.lib gdi32.lib winmm.lib kernel32.lib
::Zi is for debug.
cl /Ox /I..\include /I..\src /I..\libs ..\src\_compiler_sources.c /link /SUBSYSTEM:windows user32.lib gdi32.lib winmm.lib kernel32.lib

popd
