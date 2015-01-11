#ifndef SMITH_WATERMAN
#define SMITH_WATERMAN

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>

#include "Params.h"
#include "bit_array.h"


struct Point{
  int x;
  int y;

  Point(int x=0, int y=0){
    this -> x = x;
    this -> y = y;
  }
};



struct Score : Point{
  long value;

  Score(unsigned long value, int x=0, int y=0) : Point(x, y){
    this -> value = value;
  }
};

struct Path{
  long value;

  std::string result_line1;
  std::string result_line2;

  std::vector<Point> points;

  Path(){
    this -> value = 0;
    this -> result_line1 = "";
    this -> result_line2 = "";
  }

  void add(int x, int y){
    Point point(x, y);
    points.push_back(point);
  }
};



class SmithWaterman {
    public:
         SmithWaterman (Params &params);
        virtual ~SmithWaterman ();
        virtual void search();
        void print();

    private:
        void find_path();
        void make_path(Score &score);
        void make_result();

        unsigned int m_size_y;
        unsigned int m_size_x;

        static const int MARK_MATCH     = 1;
        static const int MARK_DELETION  = 2;
        static const int MARK_INSERTION = 3;

        Sequence m_sequence1;
        Sequence m_sequence2;

        int m_threads_count;

        long m_best_score;
        long m_best_path_value;

        std::string m_result;

        std::vector<BitArray> m_directions;

        Path * m_best_path;
        std::vector<Path> m_paths;
        CUDA_params m_cudaParams;
        std::vector<Score> m_all_scores;
        float m_duration;
};
#endif
