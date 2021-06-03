import os
import sys
import pathlib
import string


def main(args):
	if len(args) != 2:
		print("Call as: python sealFace.py YOURFOLDERNAME")
		return
	
	directory = SEALROOT#args[1] 
	#fix this in arg call, even for testing. Should be $ROOT_PATH+/path/to/dir
	
	arguments = ""
	for filename in os.listdir(directory):
		if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG"): 
			arguments += (filename + " ")
	chipfolder = directory+"Chips"
	try:
		os.mkdir(chipfolder)
	except:
		pass

	if str(os.name) == "posix":
		print("MacOS")
		os.system("./seal seal.dat " + directory + " " + arguments)
	if str(os.name) == 'nt':
		print('Window')
		os.system("seal.exe seal.dat " + directory + " " + arguments)

if __name__ == "__main__":		
	SEALROOT = str(pathlib.Path('data'))
	print(SEALROOT)
	main(sys.argv)
	#main(SEALROOT)