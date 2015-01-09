#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <new>
#include <algorithm>
#include <climits>
#include <unistd.h>
#include <cstring>

#include "Swimd.h"

using namespace std;

void fillRandomly(unsigned char* seq, int seqLength, int alphabetLength);
int * createSimpleScoreMatrix(int alphabetLength, int match, int mismatch);
int calculateSW(unsigned char query[], int queryLength, unsigned char ** db, int dbLength, int dbSeqLengths[],
                int gapOpen, int gapExt, int * scoreMatrix, int alphabetLength, SwimdSearchResult results[]);
int calculateGlobal(unsigned char query[], int queryLength, unsigned char ** db, int dbLength,
                    int dbSeqLengths[], int gapOpen, int gapExt, int * scoreMatrix,
                    int alphabetLength, SwimdSearchResult results[], const int);
void printInts(int a[], int aLength);
int maximumScore(SwimdSearchResult results[], int resultsLength);

int main(int argc, char * const argv[]) {

    char mode[32] = "SW"; // "SW", "NW", "HW" or "OV"
    if (argc > 1) {
        strcpy(mode, argv[1]);
    }

    clock_t start, finish;
    srand(42);

    int alphabetLength = 4;
    int gapOpen = 11;
    int gapExt = 1;

    // Create random query
    int queryLength = 1000;
    unsigned char query[queryLength];
    fillRandomly(query, queryLength, alphabetLength);

    // Create random database
    int dbLength = 200;
    unsigned char * db[dbLength];
    int dbSeqsLengths[dbLength];
    for (int i = 0; i < dbLength; i++) {
        dbSeqsLengths[i] = 800 + rand() % 4000;
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
#ifdef __AVX2__
    printf("Using AVX2!\n");
#elif __SSE4_1__
    printf("Using SSE4.1!\n");
#endif
    start = clock();
    SwimdSearchResult results[dbLength];
    int resultCode;
    int modeCode;
    if (!strcmp(mode, "NW")) modeCode = SWIMD_MODE_NW;
    else if (!strcmp(mode, "HW")) modeCode = SWIMD_MODE_HW;
    else if (!strcmp(mode, "OV")) modeCode = SWIMD_MODE_OV;
    else if (!strcmp(mode, "SW")) modeCode = SWIMD_MODE_SW;
    else {
        printf("Invalid mode!\n");
        return 1;
    }
    resultCode = swimdSearchDatabase(query, queryLength, db, dbLength, dbSeqsLengths,
                                     gapOpen, gapExt, scoreMatrix, alphabetLength, results,
                                     modeCode, SWIMD_OVERFLOW_SIMPLE);
    finish = clock();
    double time1 = ((double)(finish-start)) / CLOCKS_PER_SEC;

    if (resultCode == SWIMD_ERR_OVERFLOW) {
        printf("Error: overflow happened!\n");
        exit(0);
    }
    if (resultCode == SWIMD_ERR_NO_SIMD_SUPPORT) {
        printf("Error: no SIMD support!\n");
        exit(0);
    }

    printf("Time: %lf\n", time1);
    printf("Maximum: %d\n", maximumScore(results, dbLength));
    printf("\n");

    // Run normal SW
    printf("Starting normal!\n");
    start = clock();
    SwimdSearchResult results2[dbLength];
    if (!strcmp(mode, "SW")) {
        resultCode = calculateSW(query, queryLength, db, dbLength, dbSeqsLengths,
                                 gapOpen, gapExt, scoreMatrix, alphabetLength, results2);
    } else {
        resultCode = calculateGlobal(query, queryLength, db, dbLength, dbSeqsLengths,
                                     gapOpen, gapExt, scoreMatrix, alphabetLength, results2, modeCode);
    }
    finish = clock();
    double time2 = ((double)(finish-start))/CLOCKS_PER_SEC;
    printf("Time: %lf\n", time2);

    printf("Maximum: %d\n", maximumScore(results2, dbLength));
    printf("\n");

    // Print differences in scores (hopefully there are none!)
    for (int i = 0; i < dbLength; i++) {
        if (results[i].score != results2[i].score) {
            printf("#%d: score is %d but should be %d\n", i, results[i].score, results2[i].score);
        }
        if (results[i].endLocationTarget != results2[i].endLocationTarget) {
            printf("#%d: end location in target is %d but should be %d\n",
                   i, results[i].endLocationTarget, results2[i].endLocationTarget);
        }
        if (results[i].endLocationQuery != results2[i].endLocationQuery) {
            printf("#%d: end location in query is %d but should be %d\n",
                   i, results[i].endLocationQuery, results2[i].endLocationQuery);
        }
        //printf("#%d: score -> %d, end location -> (q: %d, t: %d)\n",
        //       i, results[i].score, results[i].endLocationQuery, results[i].endLocationTarget);
    }

    printf("Times faster: %lf\n", time2/time1);

    // Free allocated memory
    for (int i = 0; i < dbLength; i++) {
        delete[] db[i];
    }
    delete[] scoreMatrix;
}

void fillRandomly(unsigned char* seq, int seqLength, int alphabetLength) {
    for (int i = 0; i < seqLength; i++)
        seq[i] = rand() % alphabetLength;
}

int* createSimpleScoreMatrix(int alphabetLength, int match, int mismatch) {
    int * scoreMatrix = new int[alphabetLength*alphabetLength];
    for (int i = 0; i < alphabetLength; i++) {
        for (int j = 0; j < alphabetLength; j++)
            scoreMatrix[i * alphabetLength + j] = (i==j ? match : mismatch);
    }
    return scoreMatrix;
}

int calculateSW(unsigned char query[], int queryLength, unsigned char ** db, int dbLength,
                int dbSeqLengths[], int gapOpen, int gapExt, int * scoreMatrix, int alphabetLength,
                SwimdSearchResult results[]) {
    int prevHs[queryLength];
    int prevEs[queryLength];

    for (int seqIdx = 0; seqIdx < dbLength; seqIdx++) {
        int maxH = 0;
        int endLocationTarget = 0;
        int endLocationQuery = 0;

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
                if (H > maxH) {
                    endLocationTarget = c;
                    endLocationQuery = r;
                    maxH = H;
                }
                uF = F;
                uH = H;
                ulH = prevHs[r];

                prevHs[r] = H;
                prevEs[r] = E;
            }
        }

        results[seqIdx].score = maxH;
        results[seqIdx].endLocationTarget = endLocationTarget;
        results[seqIdx].endLocationQuery = endLocationQuery;
    }

    return 0;
}

int calculateGlobal(unsigned char query[], int queryLength, unsigned char ** db, int dbLength,
                    int dbSeqLengths[], int gapOpen, int gapExt, int * scoreMatrix,
                    int alphabetLength, SwimdSearchResult results[], const int mode) {
    int prevHs[queryLength];
    int prevEs[queryLength];

    const int LOWER_SCORE_BOUND = INT_MIN + gapExt;

    for (int seqIdx = 0; seqIdx < dbLength; seqIdx++) {
        // Initialize all values to 0
        for (int r = 0; r < queryLength; r++) {
            // Query has fixed start and end if not OV
            prevHs[r] = mode == SWIMD_MODE_OV ? 0 : -1 * gapOpen - r * gapExt;
            prevEs[r] = LOWER_SCORE_BOUND;
        }

        int maxH = INT_MIN;
        int endLocationTarget = 0;
        int endLocationQuery = 0;

        int H = INT_MIN;
        int targetLength = dbSeqLengths[seqIdx];
        for (int c = 0; c < targetLength; c++) {
            int uF, uH, ulH;
            uF = LOWER_SCORE_BOUND;
            if (mode == SWIMD_MODE_NW) { // Database sequence has fixed start and end only in NW
                uH = -1 * gapOpen - c * gapExt;
                ulH = uH + gapExt;
            } else {
                uH = ulH = 0;
            }
            if (c == 0)
                ulH = 0;

            for (int r = 0; r < queryLength; r++) {
                int E = max(prevHs[r] - gapOpen, prevEs[r] - gapExt);
                int F = max(uH - gapOpen, uF - gapExt);
                int score = scoreMatrix[query[r] * alphabetLength + db[seqIdx][c]];
                H = max(E, max(F, ulH+score));

                if (mode == SWIMD_MODE_OV && c == targetLength - 1) {
                    if (H > maxH) {
                        maxH = H;
                        endLocationQuery = r;
                        endLocationTarget = c;
                    }
                }

                uF = F;
                uH = H;
                ulH = prevHs[r];

                prevHs[r] = H;
                prevEs[r] = E;
            }

            if (H > maxH) {
                maxH = H;
                endLocationTarget = c;
                endLocationQuery = queryLength - 1;
            }
        }

        if (mode == SWIMD_MODE_NW) {
            results[seqIdx].score = H;
            results[seqIdx].endLocationTarget = targetLength - 1;
            results[seqIdx].endLocationQuery = queryLength - 1;
        } else if (mode == SWIMD_MODE_OV || mode == SWIMD_MODE_HW) {
            results[seqIdx].score = maxH;
            results[seqIdx].endLocationTarget = endLocationTarget;
            results[seqIdx].endLocationQuery = endLocationQuery;
        } else return 1;
    }

    return 0;
}

void printInts(int a[], int aLength) {
    for (int i = 0; i < aLength; i++)
        printf("%d ", a[i]);
    printf("\n");
}

int maximumScore(SwimdSearchResult results[], int resultsLength) {
    int maximum = 0;
    for (int i = 0; i < resultsLength; i++)
        if (results[i].score > maximum)
            maximum = results[i].score;
    return maximum;
}
