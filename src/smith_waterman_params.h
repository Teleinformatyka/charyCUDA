// source https://github.com/ondra-m/smith-workspace/smith-waterman/CUDA_run

#include <cstring>
#include <cstdio>


struct Sequence {
    char * data;
    int size;

    Sequence() {
        data  = NULL;
        size = 0;
    }

    ~Sequence() {
        if (size && data) {
            delete data;
            data = NULL;
            size = 0;
        }
    }
    char& operator[](int x) {
        return data[x];
    }
    Sequence& operator=(Sequence& other) {
        if (data && size) {
            delete data;
        }
        size = (other.size);
        data = new char[size];
        memcpy(data, other.data, other.size);
        return *this;


    }
};

struct Column{
    long * current;
    long * prev;
    long * before_prev;
    int size;
};

struct CUDA{
    Column column;

    char * sequence_1;
    char * sequence_2;
    char * directions;

    int iteration;
    int rows_count;
    int columns_count;
    int match;
    int mismatch;
    int gap_penalty;

    int threads_count;
    int blocks_count;
    int threads_per_block;
    int cells_per_thread;
};

struct Result{
    char * directions;
    long * column;
};

struct CUDA_params{
    CUDA cuda;

    Sequence * sequence_1;
    Sequence *  sequence_2;

    Result result;

    int column_size;
    int directions_size;
};
