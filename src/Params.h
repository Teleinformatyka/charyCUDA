#ifndef PARAMS_H
#define PARAMS_H
class Params {
    public:
        Params ();
        virtual ~Params ();
        bool parse(int argc, char**argv);


    private:
        void printUsage();
        // file with data
        char * m_dbFile;
        int m_dbSize;
        // file with query
        char * m_queryFile;
        int m_querySize;
};


#endif
