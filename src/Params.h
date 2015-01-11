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
        static int match;
        static int mismatch;
        static int gapPenalty;
        static int charPerRow;


    private:
        void printUsage();
        Sequence m_sequence1;
        Sequence m_sequence2;

};


#endif
