extern "C" {
#include <smmintrin.h>
}

#include "Swimd.hpp"

using namespace std;

#define SCORE_T short // Score type
#define UPPER_BOUND SHRT_MAX // Maximum value of score type
#define ALL1 0xFFFF
#define NUM_SEQS 8 // Number of sequences that are calculated concurrently
#define _MM_SET1 _mm_set1_epi16 // Sets all fields to given number
#define _MM_MAX _mm_max_epi16
#define _MM_ADDS _mm_adds_epi16
#define _MM_SUBS _mm_subs_epi16

/*
#define SCORE_T char // Score type
#define UPPER_BOUND CHAR_MAX // Maximum value of score type
#define ALL1 0xFF
#define NUM_SEQS 16 // Number of sequences that are calculated concurrently
#define _MM_SET1 _mm_set1_epi8 // Sets all fields to given number
#define _MM_MAX _mm_max_epi8
#define _MM_ADDS _mm_adds_epi8
#define _MM_SUBS _mm_subs_epi8
*/
/*
#define SCORE_T int // Score type
#define UPPER_BOUND INT_MAX // Maximum value of score type
#define ALL1 0xFFFFFFFF
#define NUM_SEQS 4 // Number of sequences that are calculated concurrently
#define _MM_SET1 _mm_set1_epi32 // Sets all fields to given number
#define _MM_MAX _mm_max_epi32
#define _MM_ADDS _mm_adds_epi32  // This does not exist :(
#define _MM_SUBS _mm_subs_epi32  // This does not exist :(
*/

vector<int> Swimd::searchDatabase(Byte query[], int queryLength, Byte ** db, int dbLength, int dbSeqLengths[],
				  int gapOpen, int gapExt, short ** scoreMatrix, int alphabetLength) {
    vector<int> bestScores(dbLength); // result

    int nextDbSeqIdx = 0; // index in db
    int currDbSeqsIdxs[NUM_SEQS]; // index in db
    Byte* currDbSeqsPos[NUM_SEQS]; // current element for each current database sequence
    int currDbSeqsLengths[NUM_SEQS];
    int shortestDbSeqLength = -1;  // length of shortest sequence among current database sequences
    int numEndedDbSeqs = 0; // Number of sequences that ended

    // Load initial sequences
    for (int i = 0; i < NUM_SEQS; i++)
	if (loadNextSequence(nextDbSeqIdx, dbLength, currDbSeqsIdxs[i], currDbSeqsPos[i],
			     currDbSeqsLengths[i], db, dbSeqLengths)) {
	    // Update shortest sequence length if new sequence was loaded
	    if (shortestDbSeqLength == -1 || currDbSeqsLengths[i] < shortestDbSeqLength)
		shortestDbSeqLength = currDbSeqsLengths[i];
	}

    // Q is gap open penalty, R is gap ext penalty.
    __m128i Q = _MM_SET1(gapOpen+gapExt);
    __m128i R = _MM_SET1(gapExt);

    // Previous H column (array), previous E column (array), previous F, all signed short
    __m128i prevHs[queryLength];
    __m128i prevEs[queryLength];
    // Initialize all values to 0
    for (int i = 0; i < queryLength; i++) {
	prevHs[i] = prevEs[i] = _MM_SET1(0);
    }

    __m128i maxH = _MM_SET1(0);  // Best score in sequence


    int columnsSinceLastSeqEnd = 0;
    // For each column
    while (numEndedDbSeqs < dbLength) {
	// Previous cells: u - up, l - left, ul - up left
        __m128i uF, uH, ulH; 
	uF = uH = ulH = _MM_SET1(0); // F[-1, c] = H[-1, c] = H[-1, c-1] = 0
	
	// Calculate query profile (alphabet x dbQueryLetter)
	// TODO: Rognes uses pshufb here, I don't know how/why?
	__m128i P[alphabetLength];
	SCORE_T profileRow[NUM_SEQS] __attribute__((aligned(16)));
	for (Byte letter = 0; letter < alphabetLength; letter++) {
	    short* scoreMatrixRow = scoreMatrix[letter];
	    for (int i = 0; i < NUM_SEQS; i++) {
		Byte* dbSeqPos = currDbSeqsPos[i];
		if (dbSeqPos != NULL)
		    profileRow[i] = (SCORE_T)scoreMatrixRow[*dbSeqPos];
	    }
	    P[letter] = _mm_load_si128((__m128i const*)profileRow);
	}
	
	for (int r = 0; r < queryLength; r++) { // For each cell in column
	    // Calculate E = max(lH-Q, lE-R)
	    __m128i E = _MM_MAX(_MM_SUBS(prevHs[r], Q),_MM_SUBS(prevEs[r], R) );

	    // Calculate F = max(uH-Q, uF-R)
	    __m128i F = _MM_MAX(_MM_SUBS(uH, Q),_MM_SUBS(uF, R) );

	    // Calculate H
	    __m128i H = _MM_SET1(0);
    	    H = _MM_MAX(H, E);
	    H = _MM_MAX(H, F);
	    H = _MM_MAX(H, _MM_ADDS(ulH, P[query[r]])); // Saturation prevents overflow

	    maxH = _MM_MAX(maxH, H); // update best score

	    // Set uF, uH, ulH
	    uF = F;
	    uH = H;
	    ulH = prevHs[r];

	    // Update prevHs, prevEs in advance for next column -> watch out this is some tricky code
	    prevEs[r] = E;
	    prevHs[r] = H;
	}

	SCORE_T* unpackedMaxH = (SCORE_T *)&maxH;
	
	// Check if overflow happened. Since I use saturation, I check if max possible value is reached
	for (int i = 0; i < NUM_SEQS; i++)
	    if (unpackedMaxH[i] == UPPER_BOUND)
		throw OVERFLOW_EXC;

	// Check if any of sequences ended
	columnsSinceLastSeqEnd++;
	if (shortestDbSeqLength == columnsSinceLastSeqEnd) { // If at least one sequence ended
	    shortestDbSeqLength = -1;
	    SCORE_T resetMask[NUM_SEQS] __attribute__((aligned(16)));

	    for (int i = 0; i < NUM_SEQS; i++) {
		if (currDbSeqsPos[i] != NULL) { // If not null sequence
		    currDbSeqsLengths[i] -= columnsSinceLastSeqEnd;
		    if (currDbSeqsLengths[i] == 0) { // If sequence ended
			numEndedDbSeqs++;
			// Save best sequence score
			bestScores[currDbSeqsIdxs[i]] = unpackedMaxH[i];
			// Load next sequence
			loadNextSequence(nextDbSeqIdx, dbLength, currDbSeqsIdxs[i], currDbSeqsPos[i],
					 currDbSeqsLengths[i], db, dbSeqLengths);
			resetMask[i] = 0; 
		    } else {
			resetMask[i] = ALL1;  // TODO: can this be done nicer? explore this
			if (currDbSeqsPos[i] != NULL)
			    currDbSeqsPos[i]++; // If not new and not NULL, move for one element
		    }
		    // Update shortest sequence length if sequence is not null
		    if (currDbSeqsPos[i] != NULL && (shortestDbSeqLength == -1 || currDbSeqsLengths[i] < shortestDbSeqLength))
			shortestDbSeqLength = currDbSeqsLengths[i];
		}
	    }
	    // Reset prevEs, prevHs and maxH
	    __m128i resetMaskPacked = _mm_load_si128((__m128i const*)resetMask);
	    for (int i = 0; i < queryLength; i++) {
		prevEs[i] = _mm_and_si128(prevEs[i], resetMaskPacked);
		prevHs[i] = _mm_and_si128(prevHs[i], resetMaskPacked);
	    }
	    maxH = _mm_and_si128(maxH, resetMaskPacked);

	    columnsSinceLastSeqEnd = 0;
	} else { // If no sequences ended
	    // Move for one element in all sequences
	    for (int i = 0; i < NUM_SEQS; i++)
		if (currDbSeqsPos[i] != NULL)
		    currDbSeqsPos[i]++;
	}
    }

    return bestScores;
}

inline bool Swimd::loadNextSequence(int &nextDbSeqIdx, int dbLength, int &currDbSeqIdx, Byte * &currDbSeqPos, 
				    int &currDbSeqLength, Byte ** db, int dbSeqLengths[]) {
    if (nextDbSeqIdx < dbLength) { // If there is sequence to load
	currDbSeqIdx = nextDbSeqIdx;
	currDbSeqPos = db[nextDbSeqIdx];
	currDbSeqLength = dbSeqLengths[nextDbSeqIdx];
	nextDbSeqIdx++;
	return true;
    } else { // If there are no more sequences to load
	currDbSeqIdx = currDbSeqLength = -1; // Set to -1 if there are no more sequences -> we call it null sequence
	currDbSeqPos = NULL;
	return false;
    } 
}
