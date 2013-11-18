#include "Swimd.hpp"

using namespace std;

vector<short> Swimd::searchDatabase(Byte query[], int queryLength, Byte ** db, int dbLength, int dbSeqLengths[],
				    int gapOpen, int gapExt, short ** scoreMatrix, int alphabetLength) {
    vector<short> bestScores(dbLength); // result

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
    __m128i Q = _mm_set1_epi16(gapOpen+gapExt);
    __m128i R = _mm_set1_epi16(gapExt);

    // Previous H column (array), previous E column (array), previous F, all signed short
    __m128i prevHs[queryLength];
    __m128i prevEs[queryLength];
    // Initialize all values to 0
    for (int i = 0; i < queryLength; i++) {
	prevHs[i] = prevEs[i] = _mm_set1_epi16(0);
    }

    __m128i maxH = _mm_set1_epi16(0);  // Best score in sequence


    int columnsSinceLastSeqEnd = 0;
    // For each column
    while (numEndedDbSeqs < dbLength) {
	// Previous cells: u - up, l - left, ul - up left
        __m128i uF, uH, ulH; 
	uF = uH = ulH = _mm_set1_epi16(0); // F[-1, c] = H[-1, c] = H[-1, c-1] = 0
	
	// Calculate query profile (alphabet x dbQueryLetter)
	// TODO: Rognes uses pshufb here, I don't know how/why?
	__m128i P[alphabetLength];
	short profileRow[NUM_SEQS] __attribute__((aligned(16)));
	for (Byte letter = 0; letter < alphabetLength; letter++) {
	    short* scoreMatrixRow = scoreMatrix[letter];
	    for (int i = 0; i < NUM_SEQS; i++) {
		Byte* dbSeqPos = currDbSeqsPos[i];
		if (dbSeqPos != NULL)
		    profileRow[i] = scoreMatrixRow[*dbSeqPos];
	    }
	    P[letter] = _mm_load_si128((__m128i const*)profileRow);
	}
	
	for (int r = 0; r < queryLength; r++) { // For each cell in column
	    // Calculate E = max(lH-Q, lE-R)
	    __m128i E = _mm_max_epi16( _mm_subs_epi16(prevHs[r], Q), _mm_subs_epi16(prevEs[r], R) );

	    // Calculate F = max(uH-Q, uF-R)
	    __m128i F = _mm_max_epi16( _mm_subs_epi16(uH, Q), _mm_subs_epi16(uF, R) );

	    // Calculate H
	    __m128i H = _mm_set1_epi16(0);
    	    H = _mm_max_epi16(H, E);
	    H = _mm_max_epi16(H, F);
	    H = _mm_max_epi16(H, _mm_adds_epi16(ulH, P[query[r]])); // TODO: check for overflow (here or outside the loop? We dont want to use IF in loop)

	    maxH = _mm_max_epi16(maxH, H); // update best score

	    // Set uF, uH, ulH
	    uF = F;
	    uH = H;
	    ulH = prevHs[r];

	    // Update prevHs, prevEs in advance for next column -> watch out this is some tricky code
	    prevEs[r] = E;
	    prevHs[r] = H;
	}

	// Check if any of sequences ended
	columnsSinceLastSeqEnd++;
	if (shortestDbSeqLength == columnsSinceLastSeqEnd) { // If at least one sequence ended
	    shortestDbSeqLength = -1;
	    short* unpackedMaxH = (short *)&maxH;
	    short resetMask[NUM_SEQS] __attribute__((aligned(16)));

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
			resetMask[i] = 0x0000; 
		    } else {
			resetMask[i] = 0xFFFF;  // TODO: can this be done nicer? explore this
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
