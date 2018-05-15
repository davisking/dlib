# imglab  
imglab is a simple graphical tool for annotating images with object bounding
boxes and optionally their part locations.  Generally, you use it when you want
to train an object detector (e.g. a face detector) since it allows you to
easily create the needed training dataset.  

## Compiling imglab
You can compile imglab with the following commands:

    cd dlib/tools/imglab
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release
**Note**: that you may need to install CMake (www.cmake.org) for this to work.  On a unix system you can also install imglab into /usr/local/bin by running 

    sudo make install  
This will make running it more convenient.

## How to Use
Next, to use it, lets assume you have a folder of images called /tmp/images.
These images should contain examples of the objects you want to learn to
detect.  You will use the imglab tool to label these objects.  Do this by
typing the following command:

    ./imglab -c mydataset.xml /tmp/images  
This will create a file called mydataset.xml which simply lists the images in
/tmp/images.  To add bounding boxes to the objects you run:  

    ./imglab mydataset.xml  
and a window will appear showing all the images.  

### Movement  
You can use the up and down arrow keys to cycle though the images.  

### Labeling
Labelling is done via mouse, in particular, holding the shift key, left clicking, and dragging the mouse will
allow you to draw boxes around the objects you wish to detect.  You can name your labels by  

### Adding Parts to Objects  
Suppose you wish to add parts to the objects, for e.g. add features like eyebrow landmarks (i.e. parts) to a face (i.e. an object).

To do so, in the command line type:

    ./imglab mydataset.xml --parts "eyes forehead nose left_cheek right_cheek"  
where 'eyes forehead nose left_cheek right_cheek' are the hypothetical names to the parts in the objects.

When the GUI window open,  
1. First, label the object 
2. Double click on the box and right click on the positions where you wish those values to be added
3. A popup menu will show up to choose the label from  

### Deleting unwanted labels
At times you might create some unwanted labels; to delete them -- double click on the label and press DEL i.e. delete key.  

### Saving the labels
Once you finish labeling objects go to the file menu, click save, and then
close the program. This will save the object boxes back to mydataset.xml.  You
can verify this by opening the tool again with:

    ./imglab mydataset.xml  
and observing that the boxes are present.  

### Additional info
imglab can do a few additional things.  To see these run:

    imglab -h

You can also read the instructions in the About->Help menu in the imglab's GUI panel.



