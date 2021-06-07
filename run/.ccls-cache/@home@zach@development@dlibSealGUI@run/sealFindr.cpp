#include <string>
#include <iostream>
#include <filesystem>
#include <vector>
#include <iostream>
#include <fstream>

void configRead(std::string *);
int main(int argc, char **argv){
  std::string configargs[2];
  std::string dirpath, xmlFile;

  if (argc == 2){
    dirpath = std::string(argv[2]);
    xmlFile = std::string(argv[1]);
  }
  else{
    std::cout << "reading from config";
    configRead(configargs);
    dirpath = configargs[1];
    xmlFile = configargs[0];
  }
  //std::vector<std::string> files;

  std::string arguments;
  arguments.append("./seal ");
  arguments += xmlFile;
  arguments += " ";
  arguments += dirpath;
  for (const auto & entry : std::filesystem::directory_iterator(dirpath)){
    std::cout << std::filesystem::proximate(entry, dirpath) << std::endl;
    arguments += " ";
    arguments += std::filesystem::proximate(entry, dirpath);
  }
  std::cout << arguments;

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

  else std::cout << "Unable to open file";
}
