#include <cstdio>
#include <limits>
#include <cstdlib>

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
    for (int i = 0; i < dbLength; i++)
        scores[i] = -1;

    //// --- Data for tracking sequences "loaded" in simd register --- //
    int nextDbSeqIdx = 0; // index in db
    int currDbSeqsIdxs[SIMD::numSeqs]; // index in db for each current database sequence
    unsigned char* currDbSeqsPos[SIMD::numSeqs]; // current element for each current database sequence
    int currDbSeqsResiduesLeft[SIMD::numSeqs];
    int numFinishedDbSeqs = 0; // Number of sequences that are finished
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
        loadNextSequence<SIMD>(i, nextDbSeqIdx, queryLength,
                               currDbSeqsIdxs, currDbSeqsPos, currDbSeqsResiduesLeft,
                               db, dbLength, dbSeqLengths,
                               seqStates[nextDbSeqIdx],
                               prevHs, prevEs, maxH);
    // ------------------------------------------------------------------ //


    // For each column
    while (numFinishedDbSeqs < dbLength) {	
        printf("racunam query profile\n");
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
        printf("Idem u glavnu petlju\n");
        // ----------------------- CORE LOOP (ONE COLUMN) ----------------------- //
        for (int r = 0; r < queryLength; r++) { // For each cell in column
            // Calculate E = max(lH-Q, lE-R)
            __m128i E = SIMD::max(SIMD::sub(prevHs[r], Q), SIMD::sub(prevEs[r], R));

            // Calculate F = max(uH-Q, uF-R)
            __m128i F = SIMD::max(SIMD::sub(uH, Q), SIMD::sub(uF, R));

            // Calculate H
            __m128i H = SIMD::set1(0);
    	    H = SIMD::max(H, E);
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
        }
        // ---------------------------------------------------------------------- //
        printf("Radim malo predracuna.");
        typename SIMD::type* unpackedMaxH = (typename SIMD::type*)&maxH;

        bool isSeqEnded[SIMD::numSeqs]; // isSeqEnded[i] is true if corresponding sequence ended.
        for (int i = 0; i < SIMD::numSeqs; i++)
            isSeqEnded[i] = false;


        // ------------------------ OVERFLOW DETECTION -------------------------- //
        int overflowedSimdIdxs[SIMD::numSeqs]; // Indexes of simd registers that overflowed
        int numOverflowedSeqs = 0; // Number of database sequences that overflowed
        if (!SIMD::satArthm) { // When not using saturation arithmetic
            // This check is based on following assumptions: 
            //  - overflow wraps
            //  - Q, R and all scores from scoreMatrix are between LOWER_BOUND/2 and UPPER_BOUND/2 exclusive
            typename SIMD::type* unpackedMinUlH_P = (typename SIMD::type *)&minUlH_P;
            for (int i = 0; i < SIMD::numSeqs; i++)
                if (currDbSeqsPos[i] != 0 && unpackedMinUlH_P[i] <= LOWER_BOUND/2)
                    overflowedSimdIdxs[numOverflowedSeqs++] = i;
        } else { // When using saturation arithmetic
            // Since I use saturation, I check if max possible value is reached
            for (int i = 0; i < SIMD::numSeqs; i++)
                if (currDbSeqsPos[i] != 0 && unpackedMaxH[i] == UPPER_BOUND)
                    overflowedSimdIdxs[numOverflowedSeqs++] = i;
        }
        // ---------------------------------------------------------------------- //

        // ------------------------- OVERFLOW HANDLING -------------------------- //
        // Save states of overflowed database sequences.
        for (int i = 0; i < numOverflowedSeqs; i++) {
            int simdIdx = overflowedSimdIdxs[i];
            int seqIdx = currDbSeqsIdxs[simdIdx];
            // Save state of sequence.
            DbSeqSearchState* seqState = seqStates + seqIdx;
            if (!seqState->isSet)
                seqState->set(queryLength);
            for (int r = 0; r < queryLength; r++) {
                seqState->prevHs[r] = ((typename SIMD::type*)(prevHs + r))[simdIdx]; // TODO: Is this ok?
                seqState->prevEs[r] = ((typename SIMD::type*)(prevEs + r))[simdIdx];
            }
            seqState->currSeqElement = currDbSeqsPos[simdIdx];
            seqState->numResiduesLeft = currDbSeqsResiduesLeft[i];
            // Mark sequence as ended.
            isSeqEnded[simdIdx] = true;
        }
        // ---------------------------------------------------------------------- //

        // Update number of residues left and handle finished sequences
        for (int i = 0; i < SIMD::numSeqs; i++)
            if (currDbSeqsPos[i] != 0) { // If not null sequence
                currDbSeqsResiduesLeft[i]--;
                if (currDbSeqsResiduesLeft[i] == 0) { // If sequence is finished
                    // Handle finished sequence
                    numFinishedDbSeqs++;
                    scores[currDbSeqsIdxs[i]] = unpackedMaxH[i]; // Save best sequence score.
                    isSeqEnded[i] = true; // Mark sequence as ended.
                    // Unset the state
                    seqStates[currDbSeqsIdxs[i]].unset();
                }
            }
                            
        // Load new sequences on place of those that ended.
        for (int i = 0; i < SIMD::numSeqs; i++)
            if (isSeqEnded[i]) {
                loadNextSequence<SIMD>(i, nextDbSeqIdx, queryLength,
                                       currDbSeqsIdxs, currDbSeqsPos, currDbSeqsResiduesLeft,
                                       db, dbLength, dbSeqLengths,
                                       seqStates[nextDbSeqIdx],
                                       prevHs, prevEs, maxH);
            }   
        printf("Loadao sam sekvence\n");
        // Move for one place in all sequences that were not just loaded (that did not end) and are not null.
        for (int i = 0; i < SIMD::numSeqs; i++)
            if (currDbSeqsPos[i] != 0 && !isSeqEnded[i])
                currDbSeqsPos[i]++;
        printf("Idem swapati\n");
        // Swap prevHs with Hs and prevEs with Es
        __m128i* tmp;
        tmp = prevHs; prevHs = newHs; newHs = tmp;
        tmp = prevEs; prevEs = newEs; newEs = tmp;
        printf("Gotov s kolonom\n");
    }

    return 0;
}

/*
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
    }*/

template<class SIMD>
static inline bool loadNextSequence(const int endedSeqSimdIdx, int &nextSeqIdx, const int queryLength,
                                    int currDbSeqsIdxs[], unsigned char* currDbSeqsPos[], int currDbSeqsResiduesLeft[],
                                    unsigned char** db, const int dbLength, const int dbSeqLengths[],
                                    DbSeqSearchState &nextSeqState,
                                    __m128i prevHs[], __m128i prevEs[], __m128i &maxH) {
    if (nextSeqIdx < dbLength) { // If there is sequence to load
        // Set data needed to track the sequence
        currDbSeqsIdxs[endedSeqSimdIdx] = nextSeqIdx;
        if (!nextSeqState.isSet) {
            currDbSeqsPos[endedSeqSimdIdx] = db[nextSeqIdx];
            currDbSeqsResiduesLeft[endedSeqSimdIdx] = dbSeqLengths[nextSeqIdx];
        } else {
            currDbSeqsPos[endedSeqSimdIdx] = nextSeqState.currSeqElement;
            currDbSeqsResiduesLeft[endedSeqSimdIdx] = nextSeqState.numResiduesLeft;
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
            if (nextSeqState.isSet) {
                allZero[endedSeqSimdIdx] = nextSeqState.prevHs[r];
                __m128i newH = _mm_load_si128((__m128i const*)allZero);
                prevHs[r] = SIMD::add(prevHs[r], newH);
                allZero[endedSeqSimdIdx] = 0;
            }
            // Set prevEs[r][endedSeqSimIdx]
            prevEs[r] = _mm_and_si128(prevEs[r], resetMask); // Set to 0
            if (nextSeqState.isSet) {
                allZero[endedSeqSimdIdx] = nextSeqState.prevEs[r];
                __m128i newE = _mm_load_si128((__m128i const*)allZero);
                prevEs[r] = SIMD::add(prevEs[r], newE);
                allZero[endedSeqSimdIdx] = 0;
            }
            typename SIMD::type* unpackedHs = (typename SIMD::type *)&prevHs;
            typename SIMD::type* unpackedEs = (typename SIMD::type *)&prevEs;
        }

        for (int r = 0; r < queryLength; r++) {
            typename SIMD::type* unpackedHs = (typename SIMD::type *)&prevHs[r];
            typename SIMD::type* unpackedEs = (typename SIMD::type *)&prevEs[r];
            printf("Hs[%2d]: ", r);
            for (int i = 0; i < SIMD::numSeqs; i++)
                printf("%3d ", unpackedHs[i]);
            printf("\n");
            printf("Es[%2d]: ", r);
            for (int i = 0; i < SIMD::numSeqs; i++)
                printf("%3d ", unpackedEs[i]);
            printf("\n");
        }

        // Set maxH to 0.
        maxH = _mm_and_si128(maxH, resetMask);
        typename SIMD::type* unpackedMaxH = (typename SIMD::type *)&maxH;
        // ------------------------------------------------------ //

        nextSeqIdx++;
        return true;
    } else { // If there are no more sequences to load, load "null" sequence
        printf("loadam null\n");
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

    DbSeqSearchState seqStates[dbLength];
    for (int i = 0; i < dbLength; i++)
        seqStates[i] = DbSeqSearchState();

    int resultCode;
    resultCode = swimdSearchDatabase_< Simd<char> >(query, queryLength, 
                                                    db, dbLength, dbSeqLengths, 
                                                    gapOpen, gapExt, scoreMatrix, alphabetLength, 
                                                    scores, seqStates);
    if (resultCode != 0) {
	    resultCode = swimdSearchDatabase_< Simd<short> >(query, queryLength, 
                                                         db, dbLength, dbSeqLengths,
                                                         gapOpen, gapExt, scoreMatrix, alphabetLength,
                                                         scores, seqStates);
	    if (resultCode != 0) {
	        resultCode = swimdSearchDatabase_< Simd<int> >(query, queryLength, 
                                                           db, dbLength, dbSeqLengths,
                                                           gapOpen, gapExt, scoreMatrix, alphabetLength,
                                                           scores, seqStates);
        }
    }

    return resultCode;
}
