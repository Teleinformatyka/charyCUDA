#include <fstream>
#include <iostream>
#include <new>

#include "Params.h"

Params::Params() {
    m_match = 5;
    m_mismatch = 1;
    m_gapPenalty = -1;
}

void Params::printUsage() {
    std::cerr<<"charyCuda \n"
        <<"Usage: \n"
        <<"./charyCuda db_file query_file \n";
}


Params::~Params() {

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
    file.seekg (0, std::ios::end);
    m_sequence1.size = file.tellg();
    try {
        m_sequence1.data = new char[m_sequence1.size ];
    } catch (std::bad_alloc& ba ) {
        std::cerr<<ba.what()<<"\n";
        return 0;
    }

    file.seekg (0, std::ios::beg);
    file.read (m_sequence1.data, m_sequence1.size);
    file.close();

    file.open(argv[2]);
    if (!file.is_open()) {
        std::cerr<<"Cannot open file!\n";
        printUsage();
        return 0;
    }
    file.seekg (0, std::ios::end);
    m_sequence2.size = file.tellg();
    try {
        m_sequence2.data = new char[m_sequence2.size + 1];
    } catch (std::bad_alloc& ba) {
        std::cerr<<ba.what()<<"\n";
        return 0;
    }
    file.seekg (0, std::ios::beg);
    file.read (m_sequence2.data, m_sequence2.size);
    file.close();

    return 1;
}

Sequence& Params::getSequence1() {
    return m_sequence1;
}

Sequence& Params::getSequence2() {
    return m_sequence1;
}


