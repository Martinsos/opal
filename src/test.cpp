#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <new>
#include <algorithm>
#include <climits>
#include <unistd.h>
#include <cstring>

#include "opal.h"
#include "ScoreMatrix.hpp"

using namespace std;

void fillRandomly(unsigned char* seq, int seqLength, int alphabetLength);
int * createSimpleScoreMatrix(int alphabetLength, int match, int mismatch);
int calculateSW(unsigned char query[], int queryLength, unsigned char ** db, int dbLength, int dbSeqLengths[],
                int gapOpen, int gapExt, int * scoreMatrix, int alphabetLength, OpalSearchResult* results[]);
int calculateGlobal(unsigned char query[], int queryLength, unsigned char ** db, int dbLength,
                    int dbSeqLengths[], int gapOpen, int gapExt, int * scoreMatrix,
                    int alphabetLength, OpalSearchResult* results[], const int);
void printInts(int a[], int aLength);
int maximumScore(OpalSearchResult* results[], int resultsLength);
bool checkAlignment(const unsigned char* query, int queryLength, const unsigned char* target, int targetLength,
                    const OpalSearchResult result, int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength);

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
    int queryLength = 3;
    unsigned char query[] = {2, 0, 1};

    unsigned char target[] = {3, 2, 3, 3, 1};
    int dbLength = 1;
    int dbSeqsLengths[] = {5};
    unsigned char* db[1];
    db[0] = target;
    */

    // Create score matrix
    int * scoreMatrix = createSimpleScoreMatrix(alphabetLength, 3, -1);
    /*int* scoreMatrix = ScoreMatrix::getBlosum50().getMatrix();
    int alphabetLength = ScoreMatrix::getBlosum50().getAlphabetLength();
    int gapOpen = 3;
    int gapExt = 1;*/

    // Run Opal
    printf("Starting Opal!\n");
#ifdef __AVX2__
    printf("Using AVX2!\n");
#elif __SSE4_1__
    printf("Using SSE4.1!\n");
#elif __SSE2__
    printf("Using SSE2!\n");
#elif __ARM_NEON
    printf("Using ARM NEON!\n");
#endif
    start = clock();
    OpalSearchResult* results[dbLength];
    for (int i = 0; i < dbLength; i++) {
        results[i] = new OpalSearchResult;
        opalInitSearchResult(results[i]);
    }
    int resultCode;
    int modeCode;
    if (!strcmp(mode, "NW")) modeCode = OPAL_MODE_NW;
    else if (!strcmp(mode, "HW")) modeCode = OPAL_MODE_HW;
    else if (!strcmp(mode, "OV")) modeCode = OPAL_MODE_OV;
    else if (!strcmp(mode, "SW")) modeCode = OPAL_MODE_SW;
    else {
        printf("Invalid mode!\n");
        return 1;
    }
    resultCode = opalSearchDatabase(query, queryLength, db, dbLength, dbSeqsLengths,
                                    gapOpen, gapExt, scoreMatrix, alphabetLength, results,
                                    OPAL_SEARCH_ALIGNMENT, modeCode, OPAL_OVERFLOW_SIMPLE);
    finish = clock();
    double time1 = ((double)(finish-start)) / CLOCKS_PER_SEC;

    if (resultCode == OPAL_ERR_OVERFLOW) {
        printf("Error: overflow happened!\n");
        exit(0);
    }
    if (resultCode == OPAL_ERR_NO_SIMD_SUPPORT) {
        printf("Error: no SIMD support!\n");
        exit(0);
    }

    printf("Time: %lf\n", time1);
    printf("Maximum: %d\n", maximumScore(results, dbLength));
    printf("\n");

    // Run normal SW
    printf("Starting normal!\n");
    start = clock();
    OpalSearchResult* results2[dbLength];
    for (int i = 0; i < dbLength; i++) {
        results2[i] = new OpalSearchResult;
        opalInitSearchResult(results2[i]);
    }
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
        /*
        printf("Query: ");
        for (int j = 0; j < queryLength; j++)
            printf("%d ", query[j]);
        printf("\n");
        printf("Target: ");
        for (int j = 0; j < dbSeqsLengths[i]; j++)
            printf("%d ", db[i][j]);
            printf("\n");
        printf("%d", results[i]->score);
        printf(" (%d, %d)", results[i]->startLocationQuery, results[i]->startLocationTarget);
        printf(" (%d, %d)\n", results[i]->endLocationQuery, results[i]->endLocationTarget);
        */

        if (results[i]->score != results2[i]->score) {
            printf("#%d: score is %d but should be %d\n", i, results[i]->score, results2[i]->score);
        }
        if (results[i]->endLocationTarget != results2[i]->endLocationTarget) {
            printf("#%d: end location in target is %d but should be %d\n",
                   i, results[i]->endLocationTarget, results2[i]->endLocationTarget);
        }
        if (results[i]->endLocationQuery != results2[i]->endLocationQuery) {
            printf("#%d: end location in query is %d but should be %d\n",
                   i, results[i]->endLocationQuery, results2[i]->endLocationQuery);
        }
        //printf("#%d: score -> %d, end location -> (q: %d, t: %d)\n",
        //       i, results[i]->score, results[i]->endLocationQuery, results[i]->endLocationTarget);

        checkAlignment(query, queryLength, db[i], dbSeqsLengths[i],
                       *results[i], gapOpen, gapExt, scoreMatrix, alphabetLength);
    }

    printf("Times faster: %lf\n", time2/time1);

    // Free allocated memory
    for (int i = 0; i < dbLength; i++) {
        if (results[i]->alignment) free(results[i]->alignment);
        delete results[i];
        if (results2[i]->alignment) free(results2[i]->alignment);
        delete results2[i];
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
                OpalSearchResult* results[]) {
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

        if (maxH == 0) {

        }
        results[seqIdx]->score = maxH;
        if (maxH == 0) {
            results[seqIdx]->endLocationTarget = results[seqIdx]->endLocationQuery = -1;
        } else {
            results[seqIdx]->endLocationTarget = endLocationTarget;
            results[seqIdx]->endLocationQuery = endLocationQuery;
        }
    }

    return 0;
}

int calculateGlobal(unsigned char query[], int queryLength, unsigned char ** db, int dbLength,
                    int dbSeqLengths[], int gapOpen, int gapExt, int * scoreMatrix,
                    int alphabetLength, OpalSearchResult* results[], const int mode) {
    int prevHs[queryLength];
    int prevEs[queryLength];

    const int LOWER_SCORE_BOUND = INT_MIN + gapExt;

    for (int seqIdx = 0; seqIdx < dbLength; seqIdx++) {
        // Initialize all values to 0
        for (int r = 0; r < queryLength; r++) {
            // Query has fixed start and end if not OV
            prevHs[r] = mode == OPAL_MODE_OV ? 0 : -1 * gapOpen - r * gapExt;
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
            if (mode == OPAL_MODE_NW) { // Database sequence has fixed start and end only in NW
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

                if (mode == OPAL_MODE_OV && c == targetLength - 1) {
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

        if (mode == OPAL_MODE_NW) {
            results[seqIdx]->score = H;
            results[seqIdx]->endLocationTarget = targetLength - 1;
            results[seqIdx]->endLocationQuery = queryLength - 1;
        } else if (mode == OPAL_MODE_OV || mode == OPAL_MODE_HW) {
            results[seqIdx]->score = maxH;
            results[seqIdx]->endLocationTarget = endLocationTarget;
            results[seqIdx]->endLocationQuery = endLocationQuery;
        } else return 1;
    }

    return 0;
}

void printInts(int a[], int aLength) {
    for (int i = 0; i < aLength; i++)
        printf("%d ", a[i]);
    printf("\n");
}

int maximumScore(OpalSearchResult* results[], int resultsLength) {
    int maximum = 0;
    for (int i = 0; i < resultsLength; i++)
        if (results[i]->score > maximum)
            maximum = results[i]->score;
    return maximum;
}


/**
 * Checks if alignment is correct.
 */
bool checkAlignment(const unsigned char* query, int queryLength, const unsigned char* target, int targetLength,
                    const OpalSearchResult result, int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength) {
    /*
    printf("Query: ");
    for (int i = 0; i < queryLength; i++) {
        printf("%d ", query[i]);
    }
    printf("\n");
    printf("Target: ");
    for (int i = 0; i < targetLength; i++) {
        printf("%d ", target[i]);
    }
    printf("\n");

    printf("Alignment: ");
    for (int j = 0; j < result.alignmentLength; j++)
        printf("%d ", result.alignment[j]);
    printf("\n");
    printf("%d", result.score);
    printf(" (%d, %d)", result.startLocationQuery, result.startLocationTarget);
    printf(" (%d, %d)\n", result.endLocationQuery, result.endLocationTarget);
    */

    // I think about the problem of checking alignment not through matrix, but through two sequences
    // whose elements I consume with each alignment operation.
    int alignScore = 0;
    // qIdx and tIdx point to elements that are to be consumed next.
    int qIdx = result.startLocationQuery;
    int tIdx = result.startLocationTarget;
    int prevOperation = -1;
    for (int i = 0; i < result.alignmentLength; i++) {
        // If sequence with no more elements is going to be consumed, report error.
        if ((result.alignment[i] != OPAL_ALIGN_DEL && tIdx >= targetLength)
            || (result.alignment[i] != OPAL_ALIGN_INS && qIdx >= queryLength)) {
            printf("Alignment went outside of matrix! (tIdx, qIdx, i): (%d, %d, %d)\n", tIdx, qIdx, i);
            return false;
        }

        // Do a move -> consume elements from query and/or target, and update score.
        switch (result.alignment[i]) {
        case OPAL_ALIGN_MATCH:
            if (query[qIdx] != target[tIdx]) {
                printf("Should be match but is a mismatch! (tIdx, qIdx, i): (%d, %d, %d)\n", tIdx, qIdx, i);
                return false;
            }
            alignScore += scoreMatrix[query[qIdx] * alphabetLength + target[tIdx]];
            qIdx++; tIdx++; break;
        case OPAL_ALIGN_MISMATCH:
            if (query[qIdx] == target[tIdx]) {
                printf("Should be mismatch but is a match! (tIdx, qIdx, i): (%d, %d, %d)\n", tIdx, qIdx, i);
                return false;
            }
            alignScore += scoreMatrix[query[qIdx] * alphabetLength + target[tIdx]];
            qIdx++; tIdx++; break;
        case OPAL_ALIGN_DEL:
            alignScore -= (prevOperation == OPAL_ALIGN_DEL ? gapExt : gapOpen);
            qIdx++; break;
        case OPAL_ALIGN_INS:
            alignScore -= (prevOperation == OPAL_ALIGN_INS ? gapExt : gapOpen);
            tIdx++; break;
        }

        prevOperation = result.alignment[i];
    }
    if (qIdx - 1 != result.endLocationQuery || tIdx - 1 != result.endLocationTarget) {
        printf("Alignment ended at (%d, %d) instead of (%d, %d)!\n",
               qIdx - 1, tIdx - 1, result.endLocationQuery, result.endLocationTarget);
        return false;
    }
    if (alignScore != result.score) {
        printf("Wrong score in alignment! %d should be %d\n", alignScore, result.score);
        return false;
    }
    return true;
}
