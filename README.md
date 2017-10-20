# This is just for a quick PR I needed to make Cinder-Dlib work properly
At the time of writing this, dlib's generic image interface doesn't work on some of the image processing methods because they use a certain size() function and ci::Channel and ci::Surface don't have that. To correct this dlib needs to replace all of its image methods with the required generic image interfaces
