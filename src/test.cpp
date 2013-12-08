#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <new>
#include <algorithm>
#include <ctime>

#include "Swimd.h"

using namespace std;

void fillRandomly(unsigned char* seq, int seqLength, int alphabetLength);
int * createSimpleScoreMatrix(int alphabetLength, int match, int mismatch);
int calculateSW(unsigned char query[], int queryLength, unsigned char ** db, int dbLength, int dbSeqLengths[],
                int gapOpen, int gapExt, int * scoreMatrix, int alphabetLength, int scores[]);
void printInts(int a[], int aLength);
int maximum(int a[], int aLength);

int main() {
    clock_t start, finish;
    srand(time(NULL));

    int alphabetLength = 4;
    int gapOpen = 11;
    int gapExt = 1;

    // Create random query
    int queryLength = 20;
    unsigned char query[queryLength];
    fillRandomly(query, queryLength, alphabetLength);

    // Create random database
    int dbLength = 19;
    unsigned char * db[dbLength];
    int dbSeqsLengths[dbLength];
    for (int i = 0; i < dbLength; i++) {
        dbSeqsLengths[i] = 10 + rand()%20;
        db[i] = new unsigned char[dbSeqsLengths[i]];
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
    int * scoreMatrix = createSimpleScoreMatrix(alphabetLength, 3, -1);

	
    // Run Swimd
    printf("Starting Swimd!\n");
    start = clock();
    int scores[dbLength];
    int resultCode = swimdSearchDatabase(query, queryLength, db, dbLength, dbSeqsLengths, 
                                         gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
    finish = clock();
    double time1 = ((double)(finish-start))/CLOCKS_PER_SEC;
    printf("Time: %lf\n", time1);

    if (resultCode != 0) {
        printf("Overflow happened!");
        exit(0);
    }
    // Print result
    printf("Result: ");
    printInts(scores, dbLength);
    printf("Maximum: %d\n", maximum(scores, dbLength));	  


    // Run normal SW
    printf("Starting normal SW!\n");
    start = clock();
    int scores2[dbLength];
    resultCode = calculateSW(query, queryLength, db, dbLength, dbSeqsLengths, 
                             gapOpen, gapExt, scoreMatrix, alphabetLength, scores2);
    finish = clock();
    double time2 = ((double)(finish-start))/CLOCKS_PER_SEC;
    printf("Time: %lf\n", time2);

    // Print result
    printf("Result: ");
    printInts(scores2, dbLength);
    printf("Maximum: %d\n", maximum(scores2, dbLength));
	

    // Print differences in results (hopefully there are none!)
    printf("Diff: ");
    for (int i = 0; i < dbLength; i++)
        if (scores[i] != scores2[i]) {
            printf("%d (%d,%d)  ", i, scores[i], scores2[i]);
        }
    printf("\n");

    printf("Times faster: %lf\n", time2/time1);

    // Free allocated memory
    for (int i = 0; i < dbLength; i++) {
        delete[] db[i];
    }
    delete[] scoreMatrix;
}

void fillRandomly(unsigned char* seq, int seqLength, int alphabetLength) {
    for (int i = 0; i < seqLength; i++)
        seq[i] = rand() % 4;
}

int * createSimpleScoreMatrix(int alphabetLength, int match, int mismatch) {
    int * scoreMatrix = new int[alphabetLength*alphabetLength];
    for (int i = 0; i < alphabetLength; i++) {
        for (int j = 0; j < alphabetLength; j++)
            scoreMatrix[i * alphabetLength + j] = (i==j ? match : mismatch);
    }
    return scoreMatrix;
}

int calculateSW(unsigned char query[], int queryLength, unsigned char ** db, int dbLength, int dbSeqLengths[],
                int gapOpen, int gapExt, int * scoreMatrix, int alphabetLength, int scores[]) {    
    int prevHs[queryLength];
    int prevEs[queryLength];

    for (int seqIdx = 0; seqIdx < dbLength; seqIdx++) {
        int maxH = 0;
        // Initialize all values to 0
        for (int i = 0; i < queryLength; i++) {
            prevHs[i] = prevEs[i] = 0;
        }
	
        for (int c = 0; c < dbSeqLengths[seqIdx]; c++) {
            int uF, uH, ulH;
            uF = uH = ulH = 0;

            for (int r = 0; r < queryLength; r++) {
                int E = max(prevHs[r] - gapOpen, prevEs[r] - gapExt);
                int F = max(uH - gapOpen, uF - gapExt);
                int score = scoreMatrix[query[r] * alphabetLength + db[seqIdx][c]];
                int H = max(0, max(E, max(F, ulH+score)));
                maxH = max(H, maxH);
                uF = F;
                uH = H;
                ulH = prevHs[r];
		
                prevHs[r] = H;
                prevEs[r] = E;
            }
        }
	
        scores[seqIdx] = maxH;
    }

    return 0;
}

void printInts(int a[], int aLength) {
    for (int i = 0; i < aLength; i++)
        printf("%d ", a[i]);
    printf("\n");
}

int maximum(int a[], int aLength) {
    int maximum = 0;
    for (int i = 0; i < aLength; i++)
        if (a[i] > maximum)
            maximum = a[i];
    return maximum;
}
