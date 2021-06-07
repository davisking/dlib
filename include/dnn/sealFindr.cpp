#include <string>
#include <iostream>
#include <filesystem>
#include <vector>
#include <fstream>

void configRead(std::string *);
int main(int argc, char **argv){
  std::string configargs[3];
  std::string dirpath, xmlFile, model;

  //if you run system call ./program XMLFILE MODEL PATH
  if (argc >= 3){
    dirpath = std::string(argv[3]);
    model = std::string(argv[2]);
    xmlFile = std::string(argv[1]);
  }
  //if you run system call ./program
  else if (argc == 1){
    std::cout << "Reading from config" << std::endl;
    configRead(configargs);
    dirpath = configargs[2];
    model = configargs[1];
    xmlFile = configargs[0];
  } else {
    std::cout << "Run ./program to read from config\nOr ./program YOURXMLFILE.xml YOURFOLDER1 YOURFOLDER2 ..." << std::endl;
  }

  std::string arguments = "./seal " + xmlFile + " " + model + " " + dirpath;

  //adding photo files in PATH to the arguments
  for (const auto & entry : std::filesystem::directory_iterator(dirpath)){
    arguments += " ";
    arguments += std::filesystem::proximate(entry, dirpath);
  }

  //creating chipfolder with name = PATH + Chips
  std::string chippedFolder = dirpath + "Chips";
  if(std::filesystem::create_directory(chippedFolder)){
    std::cout << "Directory created. Chipped photos will be stored at " << chippedFolder << std::endl;
  }
  else{
    std::cout << "Directory already exists.\n";
  }
  
  //calling the executable to chip faces
  system(arguments.c_str());

  return 0;
}

/**
 * Read configuration from config.txt file
 */
void configRead(std::string * config){
  std::string line;
  std::ifstream configfile("config.txt");
  int counter = 0;
  if (configfile.is_open())
  {
    while ( getline (configfile,line) )
    {
      std::string arg = line.substr(line.find('=')+1);
      config[counter] = arg;
      counter++;
    }
    configfile.close();
  }

  else std::cout << "Unable to open file" << std::endl;
}
