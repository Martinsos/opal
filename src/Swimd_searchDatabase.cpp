#include <climits>
extern "C" {
#include <smmintrin.h>
}

#include "Swimd.h"

using namespace std;

// ---------------------- PRECISION SPECIFIC MACROS --------------------- //
#ifdef SSE_8 // Flag for using char as data type for SSE
#define PRECISION 8 // Precision as number of bits
#define SCORE_T char // Score type
#define UPPER_BOUND CHAR_MAX // Maximum value of score type
#define LOWER_BOUND CHAR_MIN // Minimum value of score type
#define ALL1 0xFF // Value that has all bits set to 1
#define NUM_SEQS 16 // Number of sequences that are calculated concurrently
#define _MM_SET1 _mm_set1_epi8 // Sets all fields to given number
#define _MM_MAX _mm_max_epi8
#define _MM_MIN _mm_min_epi8
 // ------ SATURATION ARITHMETIC ------- //
#define SAT_ARTHM // Defined when saturation arithmetic is being used
#define _MM_ADD _mm_adds_epi8 // Saturation arithmetic used here!
#define _MM_SUB _mm_subs_epi8 // Saturation
 // ------------------------------------ //
#endif

#ifdef SSE_16
#define PRECISION 16 
#define SCORE_T short 
#define UPPER_BOUND SHRT_MAX
#define LOWER_BOUND SHRT_MIN
#define ALL1 0xFFFF
#define NUM_SEQS 8
#define _MM_SET1 _mm_set1_epi16
#define _MM_MAX _mm_max_epi16
#define _MM_MIN _mm_min_epi16
 // ------ SATURATION ARITHMETIC ------- //
#define SAT_ARTHM
#define _MM_ADD _mm_adds_epi16 // Saturation
#define _MM_SUB _mm_subs_epi16 // Saturation
 // ------------------------------------ //
#endif

#ifdef SSE_32
#define PRECISION 32
#define SCORE_T int
#define UPPER_BOUND INT_MAX
#define LOWER_BOUND INT_MIN
#define ALL1 0xFFFFFFFF
#define NUM_SEQS 4
#define _MM_SET1 _mm_set1_epi32
#define _MM_MAX _mm_max_epi32
#define _MM_MIN _mm_min_epi32
#define _MM_ADD _mm_add_epi32  // Saturated version does not exist
#define _MM_SUB _mm_sub_epi32  // Saturated version does not exist
#endif
// ---------------------------------------------------------------------- //

static bool loadNextSequence(int &nextDbSeqIdx, int dbLength, int &currDbSeqIdx, unsigned char * &currDbSeqPos, 
			     int &currDbSeqLength, unsigned char ** db, int dbSeqLengths[]);

// Needed because of problems argument prescan has with concatenation
#define SWIMD_SEARCH_DATABASE_DEFINITION(PRECISION) SWIMD_SEARCH_DATABASE(PRECISION)

SWIMD_SEARCH_DATABASE_DEFINITION(PRECISION) {
    // ----------------------- CHECK ARGUMENTS -------------------------- //
    // Check if Q, R or scoreMatrix have values too big for used score type
    if (gapOpen < LOWER_BOUND || UPPER_BOUND < gapOpen
	|| gapExt < LOWER_BOUND || UPPER_BOUND < gapExt) {
	return 1;
    }
#ifndef SAT_ARTHM
    // These extra limits are enforced so overflow could be detected more efficiently
    if (gapOpen <= LOWER_BOUND/2 || UPPER_BOUND/2 <= gapOpen
	|| gapExt <= LOWER_BOUND/2 || UPPER_BOUND/2 <= gapExt) {
	return 1;
    }
#endif
    for (int r = 0; r < alphabetLength; r++)
	for (int c = 0; c < alphabetLength; c++) {
	    int score = scoreMatrix[r * alphabetLength + c];
	    if (score < LOWER_BOUND || UPPER_BOUND < score) {
		return 1;
	    }
#ifndef SAT_ARTHM
	    if (score <= LOWER_BOUND/2 || UPPER_BOUND/2 <= score)
		return 1;
#endif
	}	
    // ------------------------------------------------------------------ //


    // ------------------------ INITIALIZATION -------------------------- //
    for (int i = 0; i < dbLength; i++)
	scores[i] = -1;

    int nextDbSeqIdx = 0; // index in db
    int currDbSeqsIdxs[NUM_SEQS]; // index in db
    unsigned char* currDbSeqsPos[NUM_SEQS]; // current element for each current database sequence
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
    __m128i Q = _MM_SET1(gapOpen);
    __m128i R = _MM_SET1(gapExt);

    // Previous H column (array), previous E column (array), previous F, all signed short
    __m128i prevHs[queryLength];
    __m128i prevEs[queryLength];
    // Initialize all values to 0
    for (int i = 0; i < queryLength; i++) {
	prevHs[i] = prevEs[i] = _MM_SET1(0);
    }

    __m128i maxH = _MM_SET1(0);  // Best score in sequence
    // ------------------------------------------------------------------ //



    int columnsSinceLastSeqEnd = 0;
    // For each column
    while (numEndedDbSeqs < dbLength) {	
	// -------------------- CALCULATE QUERY PROFILE ------------------------- //
	// TODO: Rognes uses pshufb here, I don't know how/why?
	__m128i P[alphabetLength];
	SCORE_T profileRow[NUM_SEQS] __attribute__((aligned(16)));
	for (unsigned char letter = 0; letter < alphabetLength; letter++) {
	    int* scoreMatrixRow = scoreMatrix + letter*alphabetLength;
	    for (int i = 0; i < NUM_SEQS; i++) {
		unsigned char* dbSeqPos = currDbSeqsPos[i];
		if (dbSeqPos != 0)
		    profileRow[i] = (SCORE_T)scoreMatrixRow[*dbSeqPos];
	    }
	    P[letter] = _mm_load_si128((__m128i const*)profileRow);
	}
	// ---------------------------------------------------------------------- //
	
	// Previous cells: u - up, l - left, ul - up left
        __m128i uF, uH, ulH; 
	uF = uH = ulH = _MM_SET1(0); // F[-1, c] = H[-1, c] = H[-1, c-1] = 0

#ifndef SAT_ARTHM
	__m128i minUlH_P = _MM_SET1(0); // Used for detecting the overflow when there is no saturation arithmetic
#endif
	
	// ----------------------- CORE LOOP (ONE COLUMN) ----------------------- //
	for (int r = 0; r < queryLength; r++) { // For each cell in column
	    // Calculate E = max(lH-Q, lE-R)
	    __m128i E = _MM_MAX(_MM_SUB(prevHs[r], Q), _MM_SUB(prevEs[r], R));

	    // Calculate F = max(uH-Q, uF-R)
	    __m128i F = _MM_MAX(_MM_SUB(uH, Q), _MM_SUB(uF, R));

	    // Calculate H
	    __m128i H = _MM_SET1(0);
    	    H = _MM_MAX(H, E);
	    H = _MM_MAX(H, F);
	    __m128i ulH_P = _MM_ADD(ulH, P[query[r]]);
	    H = _MM_MAX(H, ulH_P); // Possible overflow that is to be detected

#ifndef SAT_ARTHM
	    minUlH_P = _MM_MIN(minUlH_P, ulH_P);
#endif

	    maxH = _MM_MAX(maxH, H); // update best score

	    // Set uF, uH, ulH
	    uF = F;
	    uH = H;
	    ulH = prevHs[r];

	    // Update prevHs, prevEs in advance for next column -> watch out this is some tricky code
	    prevEs[r] = E;
	    prevHs[r] = H;
	}
	// ---------------------------------------------------------------------- //

	columnsSinceLastSeqEnd++;
	SCORE_T* unpackedMaxH = (SCORE_T *)&maxH;
	
	// ------------------------ OVERFLOW DETECTION -------------------------- //
#ifndef SAT_ARTHM
	// This check is based on following assumptions: 
	//  - overflow wraps
	//  - Q, R and all scores from scoreMatrix are between LOWER_BOUND/2 and UPPER_BOUND/2 exclusive
	SCORE_T* unpackedMinUlH_P = (SCORE_T *)&minUlH_P;
	for (int i = 0; i < NUM_SEQS; i++)
	    if (currDbSeqsPos[i] != 0 && unpackedMinUlH_P[i] <= LOWER_BOUND/2)
		return 1;
#endif
#ifdef SAT_ARTHM
	// Since I use saturation, I check if max possible value is reached
	for (int i = 0; i < NUM_SEQS; i++)
	    if (currDbSeqsPos[i] != 0 && unpackedMaxH[i] == UPPER_BOUND)
		return 1;
#endif
	// ---------------------------------------------------------------------- //

	// --------------------- CHECK AND HANDLE SEQUENCE END ------------------ //
	if (shortestDbSeqLength == columnsSinceLastSeqEnd) { // If at least one sequence ended
	    shortestDbSeqLength = -1;
	    SCORE_T resetMask[NUM_SEQS] __attribute__((aligned(16)));

	    for (int i = 0; i < NUM_SEQS; i++) {
		if (currDbSeqsPos[i] != 0) { // If not null sequence
		    currDbSeqsLengths[i] -= columnsSinceLastSeqEnd;
		    if (currDbSeqsLengths[i] == 0) { // If sequence ended
			numEndedDbSeqs++;
			// Save best sequence score
			scores[currDbSeqsIdxs[i]] = unpackedMaxH[i];
			// Load next sequence
			loadNextSequence(nextDbSeqIdx, dbLength, currDbSeqsIdxs[i], currDbSeqsPos[i],
					 currDbSeqsLengths[i], db, dbSeqLengths);
			resetMask[i] = 0; 
		    } else {
			resetMask[i] = ALL1;  // TODO: can this be done nicer? explore this
			if (currDbSeqsPos[i] != 0)
			    currDbSeqsPos[i]++; // If not new and not null, move for one element
		    }
		    // Update shortest sequence length if sequence is not null
		    if (currDbSeqsPos[i] != 0 && (shortestDbSeqLength == -1 || currDbSeqsLengths[i] < shortestDbSeqLength))
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
		if (currDbSeqsPos[i] != 0)
		    currDbSeqsPos[i]++;
	}
	// ---------------------------------------------------------------------- //
    }

    return 0;
}


static inline bool loadNextSequence(int &nextDbSeqIdx, int dbLength, int &currDbSeqIdx, unsigned char* &currDbSeqPos, 
				    int &currDbSeqLength, unsigned char** db, int dbSeqLengths[]) {
    if (nextDbSeqIdx < dbLength) { // If there is sequence to load
	currDbSeqIdx = nextDbSeqIdx;
	currDbSeqPos = db[nextDbSeqIdx];
	currDbSeqLength = dbSeqLengths[nextDbSeqIdx];
	nextDbSeqIdx++;
	return true;
    } else { // If there are no more sequences to load, load "null" sequence
	currDbSeqIdx = currDbSeqLength = -1; // Set to -1 if there are no more sequences
	currDbSeqPos = 0;
	return false;
    }
}
