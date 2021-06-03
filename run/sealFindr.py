import os
import sys
import pathlib
import string


def main(args):
	if len(args) != 2:
		print("Call as: python sealFace.py YOURFOLDERNAME")
		return
	
	directory = SEALROOT#args[1] 
	#fix this in arg call, no hardcode bullshit, even for testing. Should be $ROOT_PATH+/path/to/dir
	
	arguments = ""
	for filename in os.listdir(directory):
		if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG"): 
			arguments += (directory + "/" + filename + " ")
	chipfolder = directory+"Chips"
	try:
		os.mkdir(chipfolder)
	except:
		pass
	os.system("seal.exe seal.dat " + chipfolder + " " + arguments)

if __name__ == "__main__":		
	SEALROOT = str(pathlib.Path('data'))
	print(SEALROOT)
	main(sys.argv)
	#main(SEALROOT)
