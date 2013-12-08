#include <cstdio>
#include <limits>
#include <cstdlib>
#include <ctime>

extern "C" {
#include <immintrin.h> // AVX
}

#include "Swimd.h"

//------------------------------------ SIMD PARAMETERS ---------------------------------//
/**
 * Contains parameters and SIMD instructions specific for certain score type.
 */
template<typename T> class Simd {};

template<>
struct Simd<char> {
    typedef char type; //!< Type that will be used for score
    static const int numSeqs = 16; //!< Number of sequences that can be done in parallel.
    static const bool satArthm = true; //!< True if saturation arithmetic is used, false otherwise.
    static inline __m128i add(const __m128i& a, const __m128i& b) { return _mm_adds_epi8(a, b); }
    static inline __m128i sub(const __m128i& a, const __m128i& b) { return _mm_subs_epi8(a, b); }
    static inline __m128i min(const __m128i& a, const __m128i& b) { return _mm_min_epi8(a, b); }
    static inline __m128i max(const __m128i& a, const __m128i& b) { return _mm_max_epi8(a, b); }
    static inline __m128i set1(int a) { return _mm_set1_epi8(a); }
};

template<>
struct Simd<short> {
    typedef short type;
    static const int numSeqs = 8;
    static const bool satArthm = true;
    static inline __m128i add(const __m128i& a, const __m128i& b) { return _mm_adds_epi16(a, b); }
    static inline __m128i sub(const __m128i& a, const __m128i& b) { return _mm_subs_epi16(a, b); }
    static inline __m128i min(const __m128i& a, const __m128i& b) { return _mm_min_epi16(a, b); }
    static inline __m128i max(const __m128i& a, const __m128i& b) { return _mm_max_epi16(a, b); }
    static inline __m128i set1(int a) { return _mm_set1_epi16(a); }
};

template<>
struct Simd<int> {
    typedef int type;
    static const int numSeqs = 4;
    static const bool satArthm = false;
    static inline __m128i add(const __m128i& a, const __m128i& b) { return _mm_add_epi32(a, b); }
    static inline __m128i sub(const __m128i& a, const __m128i& b) { return _mm_sub_epi32(a, b); }
    static inline __m128i min(const __m128i& a, const __m128i& b) { return _mm_min_epi32(a, b); }
    static inline __m128i max(const __m128i& a, const __m128i& b) { return _mm_max_epi32(a, b); }
    static inline __m128i set1(int a) { return _mm_set1_epi32(a); }
};
//--------------------------------------------------------------------------------------//

/**
 * Snapshot of database sequence in one moment of database search.
 * Contains all data needed to continue search from where it stopped.
 */
struct DbSeqSearchState {
    int* prevHs; // Has length of query.
    int* prevEs; // Has length of query.
    unsigned char* currSeqElement;
    int numResiduesLeft;
    bool isSet; // True if contents are set, false if state is unset.

    DbSeqSearchState() {
        this->isSet = false;
        this->prevHs = 0;
        this->prevEs = 0;
    }
    ~DbSeqSearchState() {
        this->unset();
    }

    void set(int queryLength) {
        if (this->prevHs == 0) this->prevHs = (int*) malloc(queryLength * sizeof(int)); 
        if (this->prevEs == 0) this->prevEs = (int*) malloc(queryLength * sizeof(int));
        this->isSet = true;
    }

    void unset() {
        if (this->prevHs != 0) {
            free(this->prevHs);
            this->prevHs = 0;
        }
        if (this->prevEs != 0) {
            free(this->prevEs);
            this->prevEs = 0;
        }
        this->isSet = false;
    }
};


template<class SIMD>
static inline bool loadNextSequence(const int endedSeqSimdIdx, int &nextSeqIdx, const int queryLength,
                                    int currDbSeqsIdxs[], unsigned char* currDbSeqsPos[], int currDbSeqsResiduesLeft[],
                                    unsigned char** db, const int dbLength, const int dbSeqLengths[],
                                    DbSeqSearchState &nextSeqState,
                                    __m128i prevHs[], __m128i prevEs[], __m128i &maxH);


template<class SIMD>
static int swimdSearchDatabase_(unsigned char query[], int queryLength, 
                                unsigned char** db, int dbLength, int dbSeqLengths[],
                                int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                                int scores[], DbSeqSearchState seqStates[]) {

    static const typename SIMD::type LOWER_BOUND = std::numeric_limits<typename SIMD::type>::min();
    static const typename SIMD::type UPPER_BOUND = std::numeric_limits<typename SIMD::type>::max();

    // ----------------------- CHECK ARGUMENTS -------------------------- //
    // Check if Q or R have values too big for used score type.
    if (gapOpen < LOWER_BOUND || UPPER_BOUND < gapOpen || gapExt < LOWER_BOUND || UPPER_BOUND < gapExt) {
        return 1;
    }
    if (!SIMD::satArthm) {
        // These extra limits are enforced so overflow could be detected more efficiently
        if (gapOpen <= LOWER_BOUND/2 || UPPER_BOUND/2 <= gapOpen || gapExt <= LOWER_BOUND/2 || UPPER_BOUND/2 <= gapExt) {
            return 1;
        }
    }
    // Check if scoreMatrix has values too big for used score type.
    for (int r = 0; r < alphabetLength; r++)
        for (int c = 0; c < alphabetLength; c++) {
            int score = scoreMatrix[r * alphabetLength + c];
            if (score < LOWER_BOUND || UPPER_BOUND < score) {
                return 1;
            }
            if (!SIMD::satArthm) {
                if (score <= LOWER_BOUND/2 || UPPER_BOUND/2 <= score)
                    return 1;
            }
        }	
    // ------------------------------------------------------------------ //


    // ------------------------ INITIALIZATION -------------------------- //
    int numOverflowedSeqs = 0; // Number of overflowed database sequences.

    int numSeqsToCalculate = 0; // Number of sequences to be calculated.
    for (int i = 0; i < dbLength; i++)
        if(scores[i] == SWIMD_SCORE_UNKNOWN)
            numSeqsToCalculate++;

    __m128i zeroes = SIMD::set1(0); // Useful

    //// --- Data for tracking sequences "loaded" in simd register --- //
    int nextDbSeqIdx = 0; // index in db
    int currDbSeqsIdxs[SIMD::numSeqs]; // index in db for each current database sequence
    unsigned char* currDbSeqsPos[SIMD::numSeqs]; // current element for each current database sequence
    int currDbSeqsResiduesLeft[SIMD::numSeqs];
    int numEndedDbSeqs = 0; // Number of sequences that ended
    //// ------------------------------------------------------------- //
    
    __m128i Q = SIMD::set1(gapOpen);
    __m128i R = SIMD::set1(gapExt);
    __m128i Hs1[queryLength]; __m128i* prevHs = Hs1;
    __m128i Es1[queryLength]; __m128i* prevEs = Es1;
    __m128i Hs2[queryLength]; __m128i* newHs = Hs2;
    __m128i Es2[queryLength]; __m128i* newEs = Es2;
    __m128i maxH;  // Best score in sequence

    // Load new sequences.
    for (int i = 0; i < SIMD::numSeqs; i++)
        loadNextSequence<SIMD>(i, nextDbSeqIdx, queryLength, scores,
                               currDbSeqsIdxs, currDbSeqsPos, currDbSeqsResiduesLeft,
                               db, dbLength, dbSeqLengths,
                               seqStates,
                               prevHs, prevEs, maxH);
    // ------------------------------------------------------------------ //


    // For each column
    while (numEndedDbSeqs < numSeqsToCalculate) {
        // -------------------- CALCULATE QUERY PROFILE ------------------------- //
        // TODO: Rognes uses pshufb here, I don't know how/why?
        __m128i P[alphabetLength];
        typename SIMD::type profileRow[SIMD::numSeqs] __attribute__((aligned(16)));
        for (unsigned char letter = 0; letter < alphabetLength; letter++) {
            int* scoreMatrixRow = scoreMatrix + letter*alphabetLength;
            for (int i = 0; i < SIMD::numSeqs; i++) {
                unsigned char* dbSeqPos = currDbSeqsPos[i];
                if (dbSeqPos != 0)
                    profileRow[i] = (typename SIMD::type)scoreMatrixRow[*dbSeqPos];
            }
            P[letter] = _mm_load_si128((__m128i const*)profileRow);
        }
        // ---------------------------------------------------------------------- //
	
        // Previous cells: u - up, l - left, ul - up left
        __m128i uF, uH, ulH; 
        uF = uH = ulH = SIMD::set1(0); // F[-1, c] = H[-1, c] = H[-1, c-1] = 0

        __m128i minUlH_P = SIMD::set1(0); // Used for detecting the overflow when there is no saturation arithmetic

        // ----------------------- CORE LOOP (ONE COLUMN) ----------------------- //
        for (int r = 0; r < queryLength; r++) { // For each cell in column
            // Calculate E = max(lH-Q, lE-R)
            __m128i E = SIMD::max(SIMD::sub(prevHs[r], Q), SIMD::sub(prevEs[r], R));

            // Calculate F = max(uH-Q, uF-R)
            __m128i F = SIMD::max(SIMD::sub(uH, Q), SIMD::sub(uF, R));

            // Calculate H
            __m128i H = SIMD::max(zeroes, E);
            H = SIMD::max(H, F);
            __m128i ulH_P = SIMD::add(ulH, P[query[r]]);
            H = SIMD::max(H, ulH_P); // Possible overflow that is to be detected

            if (!SIMD::satArthm) {
                minUlH_P = SIMD::min(minUlH_P, ulH_P);
            }

            maxH = SIMD::max(maxH, H); // update best score

            // Set uF, uH, ulH
            uF = F;
            uH = H;
            ulH = prevHs[r];

            newEs[r] = E;
            newHs[r] = H;
            
            /**
             * Notice: I do not use 'set' in core loop because it is very slow!
             */
        }
        // ---------------------------------------------------------------------- //

        typename SIMD::type* unpackedMaxH = (typename SIMD::type*)&maxH;

        bool isSeqEnded[SIMD::numSeqs]; // isSeqEnded[i] is true if corresponding sequence ended.
        for (int i = 0; i < SIMD::numSeqs; i++)
            isSeqEnded[i] = false;


        // ------------------------ OVERFLOW DETECTION -------------------------- //
        int overflowedSimdIdxs[SIMD::numSeqs]; // Indexes of simd registers that overflowed
        int numOverflowedCurrSeqs = 0; // Number of current database sequences that overflowed
        if (!SIMD::satArthm) { // When not using saturation arithmetic
            // This check is based on following assumptions: 
            //  - overflow wraps
            //  - Q, R and all scores from scoreMatrix are between LOWER_BOUND/2 and UPPER_BOUND/2 exclusive
            typename SIMD::type* unpackedMinUlH_P = (typename SIMD::type *)&minUlH_P;
            for (int i = 0; i < SIMD::numSeqs; i++)
                if (currDbSeqsPos[i] != 0 && unpackedMinUlH_P[i] <= LOWER_BOUND/2)
                    overflowedSimdIdxs[numOverflowedCurrSeqs++] = i;
        } else { // When using saturation arithmetic
            // Since I use saturation, I check if max possible value is reached
            for (int i = 0; i < SIMD::numSeqs; i++)
                if (currDbSeqsPos[i] != 0 && unpackedMaxH[i] == UPPER_BOUND)
                    overflowedSimdIdxs[numOverflowedCurrSeqs++] = i;
        }
        numOverflowedSeqs += numOverflowedCurrSeqs;
        // ---------------------------------------------------------------------- //

        // ------------------------- OVERFLOW HANDLING -------------------------- //
        // Save states of overflowed database sequences.
        for (int i = 0; i < numOverflowedCurrSeqs; i++) {
            int simdIdx = overflowedSimdIdxs[i];
            int seqIdx = currDbSeqsIdxs[simdIdx];
            // Save state of sequence.
            DbSeqSearchState* seqState = seqStates + seqIdx;
            if (!seqState->isSet)
                seqState->set(queryLength);
            for (int r = 0; r < queryLength; r++) {
                typename SIMD::type* unpackedPrevHs = (typename SIMD::type *)(prevHs + r);
                typename SIMD::type* unpackedPrevEs = (typename SIMD::type *)(prevEs + r);
                seqState->prevHs[r] = unpackedPrevHs[simdIdx];
                seqState->prevEs[r] = unpackedPrevEs[simdIdx];
            }
            seqState->currSeqElement = currDbSeqsPos[simdIdx];
            seqState->numResiduesLeft = currDbSeqsResiduesLeft[simdIdx];
            // Mark sequence as ended.
            isSeqEnded[simdIdx] = true;
            numEndedDbSeqs++;            
        }
        // ---------------------------------------------------------------------- //

        // Update number of residues left and handle finished sequences
        for (int i = 0; i < SIMD::numSeqs; i++)
            if (currDbSeqsPos[i] != 0 && !isSeqEnded[i]) { // If not null sequence and not ended (by overflow)
                currDbSeqsResiduesLeft[i]--;
                if (currDbSeqsResiduesLeft[i] == 0) { // If sequence is finished
                    // Handle finished sequence
                    scores[currDbSeqsIdxs[i]] = unpackedMaxH[i]; // Save best sequence score.
                    isSeqEnded[i] = true; // Mark sequence as ended.
                    numEndedDbSeqs++;
                    // Unset the state
                    seqStates[currDbSeqsIdxs[i]].unset();
                }
            }
                            
        // Load new sequences on place of those that ended.
        for (int i = 0; i < SIMD::numSeqs; i++)
            if (isSeqEnded[i]) {
                loadNextSequence<SIMD>(i, nextDbSeqIdx, queryLength, scores,
                                       currDbSeqsIdxs, currDbSeqsPos, currDbSeqsResiduesLeft,
                                       db, dbLength, dbSeqLengths,
                                       seqStates,
                                       newHs, newEs, maxH);
            }   

        // Move for one place in all sequences that were not just loaded (that did not end) and are not null.
        for (int i = 0; i < SIMD::numSeqs; i++)
            if (currDbSeqsPos[i] != 0 && !isSeqEnded[i])
                currDbSeqsPos[i]++;

        // Swap prevHs with Hs and prevEs with Es
        __m128i* tmp;
        tmp = prevHs; prevHs = newHs; newHs = tmp;
        tmp = prevEs; prevEs = newEs; newEs = tmp;
    }

    if (numOverflowedSeqs > 0)
        return SWIMD_ERR_OVERFLOW;
    return 0;
}



template<class SIMD>
static inline bool loadNextSequence(const int endedSeqSimdIdx, int &nextSeqIdx, const int queryLength, const int scores[],
                                    int currDbSeqsIdxs[], unsigned char* currDbSeqsPos[], int currDbSeqsResiduesLeft[],
                                    unsigned char** db, const int dbLength, const int dbSeqLengths[],
                                    DbSeqSearchState seqStates[],
                                    __m128i prevHs[], __m128i prevEs[], __m128i &maxH) {
    while (scores[nextSeqIdx] != SWIMD_SCORE_UNKNOWN && nextSeqIdx < dbLength) // Skip already calculated sequences.
        nextSeqIdx++;
    if (nextSeqIdx < dbLength) { // If there is sequence to load
        DbSeqSearchState* nextSeqState = seqStates + nextSeqIdx;

        // Set data needed to track the sequence
        currDbSeqsIdxs[endedSeqSimdIdx] = nextSeqIdx;
        if (!nextSeqState->isSet) {
            currDbSeqsPos[endedSeqSimdIdx] = db[nextSeqIdx];
            currDbSeqsResiduesLeft[endedSeqSimdIdx] = dbSeqLengths[nextSeqIdx];
        } else {
            currDbSeqsPos[endedSeqSimdIdx] = nextSeqState->currSeqElement;
            currDbSeqsResiduesLeft[endedSeqSimdIdx] = nextSeqState->numResiduesLeft;
        }
        
        // ------------ Set prevHs, prevEs and maxH ------------- //
        typename SIMD::type resetMask_[SIMD::numSeqs] __attribute__((aligned(16)));
        for (int i = 0; i < SIMD::numSeqs; i++)
            resetMask_[i] = -1;
        resetMask_[endedSeqSimdIdx] = 0;
        __m128i resetMask = _mm_load_si128((__m128i const*)resetMask_);

        typename SIMD::type allZero[SIMD::numSeqs] __attribute__((aligned(16)));
        for (int i = 0; i < SIMD::numSeqs; i++)
            allZero[i] = 0;
        
        // Set prevHs and prevEs: set to 0 if no state, otherwise to value saved in state.
        for (int r = 0; r < queryLength; r++) {
            // Set prevHs[r][endedSeqSimIdx]
            prevHs[r] = _mm_and_si128(prevHs[r], resetMask); // Set to 0
            if (nextSeqState->isSet) {
                allZero[endedSeqSimdIdx] = (typename SIMD::type)nextSeqState->prevHs[r];
                prevHs[r] = SIMD::add(prevHs[r], _mm_load_si128((__m128i const*)allZero));
                allZero[endedSeqSimdIdx] = 0;
            }
            // Set prevEs[r][endedSeqSimIdx]
            prevEs[r] = _mm_and_si128(prevEs[r], resetMask); // Set to 0
            if (nextSeqState->isSet) {
                allZero[endedSeqSimdIdx] = (typename SIMD::type)nextSeqState->prevEs[r];
                prevEs[r] = SIMD::add(prevEs[r], _mm_load_si128((__m128i const*)allZero));
                allZero[endedSeqSimdIdx] = 0;
            }
        }

        // Set maxH to 0.
        maxH = _mm_and_si128(maxH, resetMask);
        // ------------------------------------------------------ //

        nextSeqIdx++;
        return true;
    } else { // If there are no more sequences to load, load "null" sequence
        currDbSeqsIdxs[endedSeqSimdIdx] = -1;
        currDbSeqsResiduesLeft[endedSeqSimdIdx] = -1;
        currDbSeqsPos[endedSeqSimdIdx] = 0; // NULL
        return false;
    }
}

extern int swimdSearchDatabase(unsigned char query[], int queryLength, 
                               unsigned char** db, int dbLength, int dbSeqLengths[],
                               int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                               int scores[]) {
    #ifndef __SSE4_1__
    return SWIMD_ERR_NO_SIMD_SUPPORT;
    #endif

    for (int i = 0; i < dbLength; i++)
        scores[i] = SWIMD_SCORE_UNKNOWN;

    DbSeqSearchState* seqStates = new DbSeqSearchState[dbLength];
    for (int i = 0; i < dbLength; i++)
        seqStates[i] = DbSeqSearchState();

    int resultCode;

    clock_t start, finish;
    start = clock();
    resultCode = swimdSearchDatabase_< Simd<char> >(query, queryLength, 
                                                    db, dbLength, dbSeqLengths, 
                                                    gapOpen, gapExt, scoreMatrix, alphabetLength, 
                                                    scores, seqStates);
    finish = clock();
    printf("Cpu time for 8bit: %lf\n", ((double)(finish-start))/CLOCKS_PER_SEC);
    if (resultCode != 0) {
        start = clock();
	    resultCode = swimdSearchDatabase_< Simd<short> >(query, queryLength, 
                                                         db, dbLength, dbSeqLengths,
                                                         gapOpen, gapExt, scoreMatrix, alphabetLength,
                                                         scores, seqStates);
        finish = clock();
        printf("Cpu time for 16bit: %lf\n", ((double)(finish-start))/CLOCKS_PER_SEC);
	    if (resultCode != 0) {
            start = clock();
	        resultCode = swimdSearchDatabase_< Simd<int> >(query, queryLength, 
                                                           db, dbLength, dbSeqLengths,
                                                           gapOpen, gapExt, scoreMatrix, alphabetLength,
                                                           scores, seqStates);
            finish = clock();
            printf("Cpu time for 32bit: %lf\n", ((double)(finish-start))/CLOCKS_PER_SEC);
        }
    }

    delete[] seqStates;
    return resultCode;
}
