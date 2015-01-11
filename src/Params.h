#ifndef PARAMS_H
#define PARAMS_H
#include "smith_waterman_params.h"

class Params {
    public:
        Params ();
        virtual ~Params ();
        bool parse(int argc, char**argv);
        Sequence& getSequence1();
        Sequence& getSequence2();


    private:
        void printUsage();
        Sequence m_sequence1;
        Sequence m_sequence2;
        int m_match;
        int m_mismatch;
        int m_gapPenalty;

};


#endif
