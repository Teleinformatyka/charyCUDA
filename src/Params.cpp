#include <fstream>
#include <iostream>
#include <new>

#include "Params.h"

int Params::match = 5;
int Params::mismatch = 1;
int Params::gapPenalty = -1;
int Params::charPerRow = 50;

Sequence Params::sequence1;
Sequence Params::sequence2;

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
    sequence1.size = file.tellg();
    try {
        sequence1.data = new char[sequence1.size ];
    } catch (std::bad_alloc& ba ) {
        std::cerr<<ba.what()<<"\n";
        return 0;
    }

    file.seekg (0, std::ios::beg);
    file.read (sequence1.data, sequence1.size);
    file.close();

    file.open(argv[2]);
    if (!file.is_open()) {
        std::cerr<<"Cannot open file!\n";
        printUsage();
        return 0;
    }
    file.seekg (0, std::ios::end);
    sequence2.size = file.tellg();
    try {
        sequence2.data = new char[sequence2.size];
    } catch (std::bad_alloc& ba) {
        std::cerr<<ba.what()<<"\n";
        return 0;
    }
    file.seekg (0, std::ios::beg);
    file.read (sequence2.data, sequence2.size);
    file.close();

    return 1;
}

Sequence& Params::getSequence1() {
    // HACK: zeby bylo szybicje
    if (sequence1.size < sequence2.size) {
        return sequence2;
    }

    return sequence1;
}

Sequence& Params::getSequence2() {
    if (sequence1.size < sequence2.size) {
        return sequence1;
    }

    return sequence2;
}


