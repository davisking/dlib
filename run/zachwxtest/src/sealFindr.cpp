#include <string>
#include <iostream>
#include <filesystem>
#include <vector>
int main(int argc, char **argv){
    std::string dirpath(argv[2]);
    std::vector<std::string> files;
    
    std::string arguments;
    std::string xmlFile(argv[1]);
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
