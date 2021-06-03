This "package" is just a copy of the stuff I use to generate the documentation 
for the dlib library.  It contains a copy of the XSLT and XML I use to 
generate the HTML documentation.

The current version of these files can be obtained from the dlib GitHub 
repository at: https://github.com/davisking/dlib

======================== Overview  ========================

I write all my documentation in XML files.  If you look through the files in 
the docs folder you will see each of them.  There is also a stylesheet.xsl 
file which contains all the XSLT I wrote to transform XML files into HTML.  
Anyway, I use that stylesheet to generate the dlib documentation from those 
XML files.  

There is also a stylesheet inside the docs/chm folder (htmlhelp_stylesheet.xsl) 
that knows how to look at the XML files and generate the table of contents 
files needed by the htmlhelp tool (the thing that makes chm help files).  

Also note that the first 80 or so lines of the stylesheet.xsl file contains
stuff specific to the dlib project and thus should be changed or removed
as appropriate if you want to reuse it for a different project. 

======================== Installing the required tools ========================

To begin with, the XML and XSLT is usable on any operating system, however, 
all the scripts I have in the docs folder that automate everything are bash 
shell scripts.  I also use stuff like wine and other Linux tools and I have 
only ever tested any of this in Debian.  So if you want to use all the scripts 
then you should probably run this stuff in Linux.  But if not you can probably 
hack something together :)

There are four scripts in the docs folder.  

 - testenv_rel: This script tests your environment for all the needed utilities.
		Run it and it should tell you what else you need to install.
		Note that the htmlify utility is something I wrote and is in
		dlib's repository in the tools/htmlify folder.  You should
		build and install it.  (go into that folder, make a subfolder
		called build, then cd into build and say:  "cmake ..; make;
		sudo make install".  You will need to install cmake if you
		don't have it already)

 - makedocs: This remakes all the HTML documentation by pulling files out
 	     of the dlib repository.  If you want to use this stuff for your
	     own projects you will need to edit this file a bit.
	     
	     Note that this script puts its output in the docs/web and
	     docs/chm/docs folders.  I use the chm folder for off-line 
	     documentation while the web folder contains what goes onto 
	     dlib.net.  Both sets of HTML are generated from the same XML 
	     files and are mostly the same.  You will see <chm></chm> and 
	     <web></web> tags inside the XML though in cases where the two 
	     differ.
	
 - makerel:  Runs makedocs as well as creates tar and zip files of the project.  
	     It also runs htmlhelp in wine to generate the chm help files.  
	     Note that you will need to run docs/chm/htmlhelp/setup_htmlhelp.sh 
	     before it will work in wine.


======================== License for documentation files ========================

To the extent possible under law, Davis E King has waived all copyright and 
related or neighboring rights to dlib documentation (XML, HTML, and XSLT files).
This work is published from United States. 

That is, I (Davis the author) don't care what you do with this.  So do
whatever you want :)



