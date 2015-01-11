#include <fstream>
#include <iostream>
#include <new>

#include "Params.h"

int Params::match = 5;
int Params::mismatch = 1;
int Params::gapPenalty = -1;
int Params::charPerRow = 4;

Params::Params() {
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
        m_sequence2.data = new char[m_sequence2.size];
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
    // HACK: zeby bylo szybicje
    if (m_sequence1.size < m_sequence2.size) {
        return m_sequence2;
    }

    return m_sequence1;
}

Sequence& Params::getSequence2() {
    if (m_sequence1.size > m_sequence2.size) {
        return m_sequence2;
    }

    return m_sequence1;
}


