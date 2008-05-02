#!/bin/sh

# Setup the registry
wine regedit htmlhelp.reg

wine regsvr32 itcc.dll
wine regsvr32 itircl.dll

