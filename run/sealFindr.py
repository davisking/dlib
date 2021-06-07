import os
import sys


def main(args):
	if len(args) != 3:
		print("Call as: python sealFindr.py YOURXMLFILE YOURFOLDERNAME")
		return
	
	directory = args[2]
	xmlFile = args[1]
	
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
		os.system("./seal " + xmlFile + " seal.dat " + directory + " " + arguments)
	if str(os.name) == 'nt':
		os.system("seal.exe " + xmlFile + " seal.dat " + directory + " " + arguments)

if __name__ == "__main__":	
    main(sys.argv)