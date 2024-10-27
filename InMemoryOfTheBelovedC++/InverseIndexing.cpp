#include "InverseIndexing.h"
#include <iostream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;
using namespace std;

void InverseIndexing::indexDocuments(const std::string& path_to_documents) {
    std::cout << path_to_documents << std::endl;
    for (const auto & entry : fs::directory_iterator(path_to_documents))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
        {
            std::ifstream myfile(entry.path());
            ostringstream oss;
            if (myfile.is_open()) {
                oss << myfile.rdbuf();
                std::string content = oss.str();
                if (!content.empty()) {
                    std::cout << content << std::endl;
                }
            } else {
                std::cout << "Could not open file: " << entry.path() << std::endl;
            }
        }
    }
    std::cout << "done" << std::endl;
}
