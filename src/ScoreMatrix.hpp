#ifndef SCORE_MATRIX_HPP
#define SCORE_MATRIX_HPP

#include <vector>

using namespace std;

class ScoreMatrix {
 private:
    vector<char> alphabet; //!< letters in same order as columns/rows in matrix
    vector<int> matrix; //!< All rows of matrix concatenated. Has length: alphabetLength * alphabetLength
   
 public:
    ScoreMatrix();
    ScoreMatrix(vector<char> alphabet, vector<int> matrix);
    ScoreMatrix(const char* filepath);
    
    int getAlphabetLength();
    char* getAlphabet();
    int* getMatrix();
    
    static ScoreMatrix getBlosum50();  

 private:
    static vector<char> getBlosumAlphabet();
};

#endif // SCORE_MATRIX_HPP
