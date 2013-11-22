#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <new>
#include <algorithm>
#include <ctime>

#include "Swimd.hpp"

using namespace std;

void fillRandomly(Byte* seq, int seqLength, int alphabetLength);
short ** createSimpleScoreMatrix(int alphabetLength, short match, short mismatch);
void deleteScoreMatrix(short ** scoreMatrix, int alphabetLength);
vector<short> calculateSW(Byte query[], int queryLength, Byte ** db, int dbLength, int dbSeqLengths[],
			  int gapOpen, int gapExt, short ** scoreMatrix, int alphabetLength);
void printShorts(vector<short> v);

int main() {
    clock_t start, finish;
    srand(time(NULL));

    int alphabetLength = 4;
    int gapOpen = 11;
    int gapExt = 1;

    // Create random query
    int queryLength = 300;
    Byte query[queryLength];
    fillRandomly(query, queryLength, alphabetLength);

    // Create random database
    int dbLength = 500;
    Byte * db[dbLength];
    int dbSeqsLengths[dbLength];
    for (int i = 0; i < dbLength; i++) {
	dbSeqsLengths[i] = 150 + rand()%200;
	db[i] = new Byte[dbSeqsLengths[i]];
	fillRandomly(db[i], dbSeqsLengths[i], alphabetLength);
    }
    
    /*
      printf("Query:\n");
      for (int i = 0; i < queryLength; i++)
      printf("%d ", query[i]);
      printf("\n");

      printf("Database:\n");
      for (int i = 0; i < dbLength; i++) {
      printf("%d. ", i); 
      for (int j = 0; j < dbSeqsLengths[i]; j++)
      printf("%d ", db[i][j]);
      printf("\n");
      }
    */

    // Create score matrix
    short ** scoreMatrix = createSimpleScoreMatrix(alphabetLength, 100, -1);

	
    // Run Swimd
    printf("Starting Swimd!\n");
    start = clock();
    vector<short> scores;
    try {
	scores = Swimd::searchDatabase(query, queryLength, db, dbLength, dbSeqsLengths, 
				      gapOpen, gapExt, scoreMatrix, alphabetLength);
    } catch (int e) {
	if (e == OVERFLOW_EXC)
	    printf("Overflow occured!");
	exit(0);
    }

    finish = clock();
    double time1 = ((double)(finish-start))/CLOCKS_PER_SEC;
    printf("Time: %lf\n", ((double)(finish-start))/CLOCKS_PER_SEC);

    // Print result
    printf("Result: ");
    printShorts(scores);
	  
    // Run normal SW
    printf("Starting normal SW!\n");
    start = clock();
    vector<short> scores2 = calculateSW(query, queryLength, db, dbLength, dbSeqsLengths, 
					gapOpen, gapExt, scoreMatrix, alphabetLength);
    finish = clock();
    double time2 = ((double)(finish-start))/CLOCKS_PER_SEC;
    printf("Time: %lf\n", ((double)(finish-start))/CLOCKS_PER_SEC);

    // Print result
    printf("Result: ");
    printShorts(scores2);
	
    // Print differences in results (hopefully there are none!)
    printf("Diff: ");
    for (int i = 0; i < scores.size(); i++)
	if (scores[i] != scores2[i]) {
	    printf("%d (%d,%d)  ", i, scores[i], scores2[i]);
	}
    printf("\n");

    printf("Times faster: %lf\n", time2/time1);

    // Free allocated memory
    for (int i = 0; i < dbLength; i++) {
	delete[] db[i];
    }
    deleteScoreMatrix(scoreMatrix, alphabetLength);
}

void fillRandomly(Byte* seq, int seqLength, int alphabetLength) {
    for (int i = 0; i < seqLength; i++)
	seq[i] = rand() % 4;
}

short ** createSimpleScoreMatrix(int alphabetLength, short match, short mismatch) {
    short ** scoreMatrix = new short*[alphabetLength];
    for (int i = 0; i < alphabetLength; i++) {
	scoreMatrix[i] = new short[alphabetLength];
	for (int j = 0; j < alphabetLength; j++)
	    scoreMatrix[i][j] = (i==j ? match : mismatch);
    }
    return scoreMatrix;
}

void deleteScoreMatrix(short ** scoreMatrix, int alphabetLength) {
    for (int i = 0; i < alphabetLength; i++)
	delete[] scoreMatrix[i];
    delete[] scoreMatrix;
}

vector<short> calculateSW(Byte query[], int queryLength, Byte ** db, int dbLength, int dbSeqLengths[],
			  int gapOpen, int gapExt, short ** scoreMatrix, int alphabetLength) {
    vector<short> bestScores(dbLength); // result
    
    short prevHs[queryLength];
    short prevEs[queryLength];

    for (int seqIdx = 0; seqIdx < dbLength; seqIdx++) {
	short maxH = 0;
	// Initialize all values to 0
	for (int i = 0; i < queryLength; i++) {
	    prevHs[i] = prevEs[i] = 0;
	}
	
	for (int c = 0; c < dbSeqLengths[seqIdx]; c++) {
	    short uF, uH, ulH;
	    uF = uH = ulH = 0;

	    for (int r = 0; r < queryLength; r++) {
		short E = max(prevHs[r] - (gapOpen + gapExt), prevEs[r] - gapExt);
		short F = max(uH - (gapOpen + gapExt), uF - gapExt);
		short score = scoreMatrix[query[r]][db[seqIdx][c]];
		short H = max((short)0, max(E, max(F, (short)(ulH+score))));
		maxH = max(H, maxH);
		uF = F;
		uH = H;
		ulH = prevHs[r];
		
		prevHs[r] = H;
		prevEs[r] = E;
	    }
	}
	
	bestScores[seqIdx] = maxH;
    }

    return bestScores;
}

void printShorts(vector<short> v) {
    for (int i = 0; i < v.size(); i++)
	printf("%d ", v[i]);
    printf("\n");
}
