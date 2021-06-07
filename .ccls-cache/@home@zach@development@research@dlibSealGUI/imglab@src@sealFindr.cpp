#include <string>
#include <iostream>
#include <filesystem>
#include <vector>
int main(int argc, char **argv){
  std::string path(argv[2]);
  std::vector<std::string> files;
  for (const auto & entry : std::filesystem::directory_iterator(path)){
    std::cout << std::filesystem::relative(path, entry.path()) << std::endl;
    //files.push_back(entry.path());
  }
  //std::cout << files.data();
  return 0;
}
