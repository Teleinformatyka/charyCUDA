
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

#include "SmithWaterman.h"
void searchCUDA(CUDA_params&);
void initCUDA(CUDA_params&);
void deinitCUDA(CUDA&);

std::string to_s(int x, int digits = 0){
   std::ostringstream tmp;


  if(digits == 0){
    tmp << x;
  }  else{
    tmp << std::setw(digits) << x;
  }

  return tmp.str();
}

SmithWaterman::SmithWaterman(Params &params)
{
    if (Params::sequence1.size > Params::sequence2.size) {
        m_size_x = Params::sequence1.size;
        m_size_y = Params::sequence2.size;

        m_sequence1 = Params::sequence1;
        m_sequence2 = Params::sequence2;
    } else {
        m_size_x = Params::sequence2.size;
        m_size_y = Params::sequence1.size;

        m_sequence1 = Params::sequence2;
        m_sequence2 = Params::sequence1;

    }

    m_cudaParams.directions_size = m_size_y;

    m_cudaParams.result.directions = new char[m_size_y];
    m_cudaParams.result.column = new long[m_size_y+1];

    m_cudaParams.sequence_1 = &m_sequence1;

    m_cudaParams.sequence_2 = &m_sequence2;


    m_cudaParams.cuda.match = Params::match;
    m_cudaParams.cuda.mismatch = Params::mismatch;
    m_cudaParams.cuda.gap_penalty = Params::gapPenalty;

    m_cudaParams.cuda.cells_per_thread = 24;
    m_cudaParams.cuda.threads_per_block = Params::threads_per_block;
    m_cudaParams.cuda.threads_count = ceil((float)m_size_y / (float)m_cudaParams.cuda.cells_per_thread);
    m_cudaParams.cuda.blocks_count = ceil((float)m_cudaParams.cuda.threads_count / (float)m_cudaParams.cuda.threads_per_block);


    m_threads_count = m_cudaParams.cuda.threads_count;

    m_directions.resize(m_size_x+1);

    for(int x=0; x<m_size_x+1; x++){
        m_directions[x].make(2, m_size_y+1, 0);
    }
    m_best_score = 0;
    m_best_path_value = 0;
    m_result = "";
    m_duration = 0;


}

SmithWaterman::~SmithWaterman() {

}

void SmithWaterman::search() {

    cudaEvent_t start, stop;

    long max_value, value;


    initCUDA(m_cudaParams);
    float duration = 0;

    max_value = 0;
    for(int iteration=0; iteration<(m_size_x+(m_size_y/m_cudaParams.cuda.threads_count)); iteration++){

        m_cudaParams.cuda.iteration = iteration;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        searchCUDA(m_cudaParams);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&duration, start, stop);
        m_duration += duration;

        value = *std::max_element(m_cudaParams.result.column, m_cudaParams.result.column+m_size_y + 1);
        if(value > max_value){
            m_all_scores.clear();
            m_best_score = max_value = value;
        }

        for(int thread=0; thread<m_cudaParams.cuda.threads_count; thread++){
            int x = iteration - thread;
            int y = thread * m_cudaParams.cuda.cells_per_thread;
            int end_y = y + m_cudaParams.cuda.cells_per_thread;

            while(y < end_y && y <= m_size_y && x >= 0 && x < m_size_x){

                m_directions[x+1].set(y+1, m_cudaParams.result.directions[y]);

                if(m_cudaParams.result.column[y+1] == max_value){
                  Score s(value, x+1, y+1);
                  m_all_scores.push_back(s);
                }

                y++;
            }
        }

    }


    deinitCUDA(m_cudaParams.cuda);
}

// -------------------------------------------------------------------------------------------

void SmithWaterman::find_path(){

    for(auto score = m_all_scores.begin(); score != m_all_scores.end(); ++score){
        make_path(*score);
    }

    for(auto path=m_paths.begin(); path!=m_paths.end(); ++path){

        if((*path).value == m_best_path_value){
            m_best_path = &(*path);
            break;
        }
    }

    make_result();
}

// -------------------------------------------------------------------------------------------
//
// Constructing m_result path
//
// x, y is +1 because of first row and column filled with zero
//
void SmithWaterman::make_path(Score &score){

  Path path;

  int x = score.x;
  int y = score.y;

  bool end = false;

  do {

    path.add(x, y);

    switch(m_directions[x][y]){
      case 0:
        end = true;
        break;

      case MARK_MATCH:
        x--;
        y--;

        path.result_line1 += m_sequence1[x];
        path.result_line2 += m_sequence2[y];

        path.value += (m_sequence1[x] == m_sequence2[y]) ? Params::match : Params::mismatch;
        break;

      case MARK_INSERTION:
        y--;

        path.result_line1 += '-';
        path.result_line2 += m_sequence2[y];

        path.value += Params::gapPenalty;
        break;

      case MARK_DELETION:
        x--;

        path.result_line1 += m_sequence1[x];
        path.result_line2 += '-';


        path.value += Params::gapPenalty;
        break;
    }

  } while(!end);

  std::reverse(path.points.begin(), path.points.end());
  std::reverse(path.result_line1.begin(), path.result_line1.end());
  std::reverse(path.result_line2.begin(), path.result_line2.end());

  if(path.value > m_best_path_value){
    m_best_path_value = path.value;
  }

  m_paths.push_back(path);
}


void SmithWaterman::make_result() {

    std::string result1;
    std::string result2;
    std::string result3;

    int biggest_number = std::max((m_best_path -> points).back().x, (m_best_path -> points).back().y);
    int digits = 0;
    do { biggest_number /= 10; digits++; } while (biggest_number != 0);

    int x = (m_best_path -> points).front().x;
    int y = (m_best_path -> points).front().y;

    int local_i = 0;

    for(int i=0; i<(m_best_path -> result_line1).size(); i++){

        if(local_i == 0){
            result1 = to_s(x, digits) + " ";
            result2 = std::string(digits+1, ' ');
            result3 = to_s(y, digits) + " ";
        }

        result1 += (m_best_path->result_line1)[i];
        result2 += ((m_best_path->result_line1)[i] == (m_best_path->result_line2)[i] ? '|' : ' ');
        result3 += (m_best_path->result_line2)[i];

        if(local_i == Params::charPerRow || (i+1) == (m_best_path->result_line1).size()){
            local_i = -1;

            result1 += " " + to_s(x, digits);
            result3 += " " + to_s(y, digits);

            m_result += result1;
            m_result += '\n';
            m_result += result2;
            m_result += '\n';
            m_result += result3;
            m_result += '\n';
            m_result += '\n';
        }

        if((m_best_path->result_line1)[i] != '-'){ x++; }
        if((m_best_path->result_line2)[i] != '-'){ y++; }

        local_i++;
    }
}


void SmithWaterman::print(){

    find_path();

    std::cout << "Match: " << Params::match << std::endl;
    std::cout << "Mismatch: " << Params::mismatch << std::endl;
    std::cout << "Gap penalty: " << Params::gapPenalty << std::endl;

    std::cout << "Size of #1: " << m_sequence1.size << std::endl;
    std::cout << "Size of #2: " << m_sequence2.size << std::endl;

    std::cout << "Finded path: " << m_paths.size() << std::endl;
    std::cout << "Time: " << m_duration << "ms" << std::endl;

    std::cout << std::endl;

    // std::cout << m_result << std::endl;
}


