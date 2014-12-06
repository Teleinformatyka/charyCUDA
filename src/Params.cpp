#include <fstream>
#include <iostream>

#include "Params.h"

Params::Params() {
    m_queryFile = nullptr;
    m_dbFile = nullptr;

}

void Params::printUsage() {
    std::cerr<<"charyCuda \n"
        <<"Usage: \n"
        <<"./charyCuda db_file query_file \n";
}


Params::~Params() {

    if (m_queryFile) {
        delete[] m_queryFile;
        delete[] m_dbFile;

    }
}

bool Params::parse(int argc, char **argv) {
    if (argc < 3) {
        printUsage();
        return 0;
    }
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr<<"Cannot open file!\n";
        printUsage();
        return 0;
    }
    m_dbSize = file.tellg();
    m_dbFile = new char[m_dbSize];
    file.seekg (0, std::ios::beg);
    file.read (m_dbFile, m_dbSize);
    file.close();

    file.open(argv[2]);
    if (!file.is_open()) {
        std::cerr<<"Cannot open file!\n";
        printUsage();
        return 0;
    }
    m_querySize = file.tellg();
    m_queryFile = new char[m_querySize];
    file.seekg (0, std::ios::beg);
    file.read (m_dbFile, m_querySize);
    file.close();

    return 1;

}
