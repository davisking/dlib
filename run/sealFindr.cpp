#include <string>
#include <iostream>
#include <filesystem>
#include <vector>
#include <iostream>
#include <fstream>

void configRead(std::string *);
int main(int argc, char **argv){
  std::string configargs[3];
  std::string dirpath, xmlFile, model;

  if (argc >= 3){
    dirpath = std::string(argv[3]);
    model = std::string(argv[2]);
    xmlFile = std::string(argv[1]);
  }
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

  for (const auto & entry : std::filesystem::directory_iterator(dirpath)){
    std::cout << std::filesystem::proximate(entry, dirpath) << std::endl;
    arguments += " ";
    arguments += std::filesystem::proximate(entry, dirpath);
  }
  std::cout << arguments;

  system(arguments.c_str());

  return 0;
}

int configWrite(){
  std::ofstream configfile;
  configfile.open("config.txt");
  return 0;
}
void configRead(std::string * config){
  std::string line;
  std::ifstream configfile("config.txt");
  int counter = 0;
  if (configfile.is_open())
  {
    while ( getline (configfile,line) )
    {
      std::string arg = line.substr(line.find('=')+1);
      //std::cout << arg;
      config[counter] = arg;
      counter++;
      //std::cout << line << '\n';
    }
    configfile.close();
  }

  else std::cout << "Unable to open file" << std::endl;
}
