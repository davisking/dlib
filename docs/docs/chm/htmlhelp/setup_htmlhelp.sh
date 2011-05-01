#!/bin/sh

cp *.dll ~/.wine/drive_c/windows/system32/

# Setup the registry
wine regedit htmlhelp.reg

wine regsvr32 itcc.dll
wine regsvr32 itircl.dll

