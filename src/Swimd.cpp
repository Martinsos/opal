#include <cstdio>
#include <limits>

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
    static const bool negRange = true; //!< True if it uses negative range for score representation, goes with saturation
    static inline __m128i add(const __m128i& a, const __m128i& b) { return _mm_adds_epi8(a, b); }
    static inline __m128i sub(const __m128i& a, const __m128i& b) { return _mm_subs_epi8(a, b); }
    static inline __m128i min(const __m128i& a, const __m128i& b) { return _mm_min_epu8(a, b); }
    static inline __m128i max(const __m128i& a, const __m128i& b) { return _mm_max_epu8(a, b); }
    static inline __m128i set1(int a) { return _mm_set1_epi8(a); }
};

template<>
struct Simd<short> {
    typedef short type;
    static const int numSeqs = 8;
    static const bool satArthm = true;
    static const bool negRange = false;
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
    static const bool negRange = false;
    static inline __m128i add(const __m128i& a, const __m128i& b) { return _mm_add_epi32(a, b); }
    static inline __m128i sub(const __m128i& a, const __m128i& b) { return _mm_sub_epi32(a, b); }
    static inline __m128i min(const __m128i& a, const __m128i& b) { return _mm_min_epi32(a, b); }
    static inline __m128i max(const __m128i& a, const __m128i& b) { return _mm_max_epi32(a, b); }
    static inline __m128i set1(int a) { return _mm_set1_epi32(a); }
};
//--------------------------------------------------------------------------------------//


static bool loadNextSequence(int &nextDbSeqIdx, int dbLength, int &currDbSeqIdx, unsigned char * &currDbSeqPos, 
                             int &currDbSeqLength, unsigned char ** db, int dbSeqLengths[]);


// For debugging
template<class SIMD>
void print_mm128i(__m128i mm) {
    typename SIMD::type* unpacked = (typename SIMD::type *)&mm;
    for (int i = 0; i < SIMD::numSeqs; i++)
        printf("%d ", unpacked[i]);
}

template<class SIMD>
static int swimdSearchDatabase_(unsigned char query[], int queryLength, 
                                unsigned char** db, int dbLength, int dbSeqLengths[],
                                int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                                int scores[]) {

    static const typename SIMD::type LOWER_BOUND = std::numeric_limits<typename SIMD::type>::min();
    static const typename SIMD::type UPPER_BOUND = std::numeric_limits<typename SIMD::type>::max();

    // ----------------------- CHECK ARGUMENTS -------------------------- //
    // Check if Q, R or scoreMatrix have values too big for used score type
    if (gapOpen < LOWER_BOUND || UPPER_BOUND < gapOpen || gapExt < LOWER_BOUND || UPPER_BOUND < gapExt) {
        return 1;
    }
    if (!SIMD::satArthm) {
        // These extra limits are enforced so overflow could be detected more efficiently
        if (gapOpen <= LOWER_BOUND/2 || UPPER_BOUND/2 <= gapOpen || gapExt <= LOWER_BOUND/2 || UPPER_BOUND/2 <= gapExt) {
            return 1;
        }
    }

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

    __m128i zeroes = SIMD::set1(0);
    __m128i scoreZeroes; // 0 normally, but lower bound if using negative range
    if (SIMD::negRange)
        scoreZeroes = SIMD::set1(LOWER_BOUND);
    else
        scoreZeroes = zeroes;

    int nextDbSeqIdx = 0; // index in db
    int currDbSeqsIdxs[SIMD::numSeqs]; // index in db
    unsigned char* currDbSeqsPos[SIMD::numSeqs]; // current element for each current database sequence
    int currDbSeqsLengths[SIMD::numSeqs];
    int shortestDbSeqLength = -1;  // length of shortest sequence among current database sequences
    int numEndedDbSeqs = 0; // Number of sequences that ended

    // Load initial sequences
    for (int i = 0; i < SIMD::numSeqs; i++)
        if (loadNextSequence(nextDbSeqIdx, dbLength, currDbSeqsIdxs[i], currDbSeqsPos[i],
                             currDbSeqsLengths[i], db, dbSeqLengths)) {
            // Update shortest sequence length if new sequence was loaded
            if (shortestDbSeqLength == -1 || currDbSeqsLengths[i] < shortestDbSeqLength)
                shortestDbSeqLength = currDbSeqsLengths[i];
        }

    // Q is gap open penalty, R is gap ext penalty.
    __m128i Q = SIMD::set1(gapOpen);
    __m128i R = SIMD::set1(gapExt);

    // Previous H column (array), previous E column (array), previous F, all signed short
    __m128i prevHs[queryLength];
    __m128i prevEs[queryLength];
    // Initialize all values to 0
    for (int i = 0; i < queryLength; i++) {
        prevHs[i] = prevEs[i] = scoreZeroes;
    }

    __m128i maxH = scoreZeroes;  // Best score in sequence
    // ------------------------------------------------------------------ //


    int columnsSinceLastSeqEnd = 0;
    // For each column
    while (numEndedDbSeqs < dbLength) {	
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
        uF = uH = ulH = scoreZeroes; // F[-1, c] = H[-1, c] = H[-1, c-1] = 0

        __m128i ofTest = scoreZeroes; // Used for detecting the overflow when not using saturated ar
	
        // ----------------------- CORE LOOP (ONE COLUMN) ----------------------- //
        for (int r = 0; r < queryLength; r++) { // For each cell in column
            // Calculate E = max(lH-Q, lE-R)
            __m128i E = SIMD::max(SIMD::sub(prevHs[r], Q), SIMD::sub(prevEs[r], R));

            // Calculate F = max(uH-Q, uF-R)
            __m128i F = SIMD::max(SIMD::sub(uH, Q), SIMD::sub(uF, R));

            // Calculate H
    	    __m128i H = SIMD::max(F, E);
            if (!SIMD::negRange) // If not using negative range, then H could be negative at this moment so we need this
                H = SIMD::max(H, zeroes);
            __m128i ulH_P = SIMD::add(ulH, P[query[r]]); // If using negative range: if ulH_P >= 0 then we have overflow

            H = SIMD::max(H, ulH_P); // If using negative range: H will always be negative, even if ulH_P overflowed

            // Save data needed for overflow detection. Not more then one condition will fire
            if (SIMD::negRange)
                ofTest = _mm_and_si128(ofTest, ulH_P);
            if (!SIMD::satArthm)
                ofTest = SIMD::min(ofTest, ulH_P);

            maxH = SIMD::max(maxH, H); // update best score

            // Set uF, uH, ulH
            uF = F;
            uH = H;
            ulH = prevHs[r];

            // Update prevHs, prevEs in advance for next column
            prevEs[r] = E;
            prevHs[r] = H;

            // For saturated: score is biased everywhere, but just score: E, F, H
            // Also, all scores except ulH_P certainly have value < 0
        }
        // ---------------------------------------------------------------------- //

        columnsSinceLastSeqEnd++;
        typename SIMD::type* unpackedMaxH = (typename SIMD::type *)&maxH;
	
        // ------------------------ OVERFLOW DETECTION -------------------------- //
        if (!SIMD::satArthm) {
            // This check is based on following assumptions: 
            //  - overflow wraps
            //  - Q, R and all scores from scoreMatrix are between LOWER_BOUND/2 and UPPER_BOUND/2 exclusive
            typename SIMD::type* unpackedOfTest = (typename SIMD::type *)&ofTest;
            for (int i = 0; i < SIMD::numSeqs; i++)
                if (currDbSeqsPos[i] != 0 && unpackedOfTest[i] <= LOWER_BOUND/2)
                    return 1;
        } else {
            if (SIMD::negRange) {
                // Since I use saturation, I check if minUlH_P was non negative
                typename SIMD::type* unpackedOfTest = (typename SIMD::type *)&ofTest;
                for (int i = 0; i < SIMD::numSeqs; i++)
                    if (currDbSeqsPos[i] != 0 && unpackedOfTest[i] >= 0)
                        return 1;
            } else {
                // I check if upper bound is reached
                for (int i = 0; i < SIMD::numSeqs; i++)
                    if (currDbSeqsPos[i] != 0 && unpackedMaxH[i] == UPPER_BOUND) {
                        return 1;
                    }
            }
        }
        // ---------------------------------------------------------------------- //

        // --------------------- CHECK AND HANDLE SEQUENCE END ------------------ //
        if (shortestDbSeqLength == columnsSinceLastSeqEnd) { // If at least one sequence ended
            shortestDbSeqLength = -1;
            typename SIMD::type resetMask[SIMD::numSeqs] __attribute__((aligned(16)));

            for (int i = 0; i < SIMD::numSeqs; i++) {
                if (currDbSeqsPos[i] != 0) { // If not null sequence
                    currDbSeqsLengths[i] -= columnsSinceLastSeqEnd;
                    if (currDbSeqsLengths[i] == 0) { // If sequence ended
                        numEndedDbSeqs++;
                        // Save best sequence score
                        scores[currDbSeqsIdxs[i]] = unpackedMaxH[i];
                        if (SIMD::negRange)
                            scores[currDbSeqsIdxs[i]] -= LOWER_BOUND;
                        // Load next sequence
                        loadNextSequence(nextDbSeqIdx, dbLength, currDbSeqsIdxs[i], currDbSeqsPos[i],
                                         currDbSeqsLengths[i], db, dbSeqLengths);
                        if (SIMD::negRange)
                            resetMask[i] = LOWER_BOUND; //Sets to LOWER_BOUND when used with saturated add and value < 0
                        else
                            resetMask[i] = 0; // Sets to zero when used with and
                    } else {
                        if (SIMD::negRange)
                            resetMask[i] = 0; // Does not change anything when used with saturated add
                        else
                            resetMask[i] = -1; // All 1s, does not change anything when used with and
                            
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
            if (SIMD::negRange) {
                for (int i = 0; i < queryLength; i++)
                    prevEs[i] = SIMD::add(prevEs[i], resetMaskPacked);
                for (int i = 0; i < queryLength; i++)
                    prevHs[i] = SIMD::add(prevHs[i], resetMaskPacked);
                maxH = SIMD::add(maxH, resetMaskPacked);
            } else {
                for (int i = 0; i < queryLength; i++)
                    prevEs[i] = _mm_and_si128(prevEs[i], resetMaskPacked);
                for (int i = 0; i < queryLength; i++)
                    prevHs[i] = _mm_and_si128(prevHs[i], resetMaskPacked);
                maxH = _mm_and_si128(maxH, resetMaskPacked);
            }
            columnsSinceLastSeqEnd = 0;
        } else { // If no sequences ended
            // Move for one element in all sequences
            for (int i = 0; i < SIMD::numSeqs; i++)
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

extern int swimdSearchDatabase(unsigned char query[], int queryLength, 
                               unsigned char** db, int dbLength, int dbSeqLengths[],
                               int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                               int scores[]) {
    #ifndef __SSE4_1__
    return SWIMD_ERR_NO_SIMD_SUPPORT;
    #endif

    // TODO: initialize scores in advance and keep the results.

    int resultCode;
    printf("Using char\n");
    resultCode = swimdSearchDatabase_< Simd<char> >(query, queryLength, 
                                                    db, dbLength, dbSeqLengths, 
                                                    gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
    if (resultCode != 0) {
        printf("Using short\n");
	    resultCode = swimdSearchDatabase_< Simd<short> >(query, queryLength, 
                                                         db, dbLength, dbSeqLengths,
                                                         gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
	    if (resultCode != 0) {
            printf("Using int\n");
	        resultCode = swimdSearchDatabase_< Simd<int> >(query, queryLength, 
                                                           db, dbLength, dbSeqLengths,
                                                           gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
        }
    }

    return resultCode;
}











//------------------------------------ SIMD PARAMETERS FOR GLOBAL ALIGNEMNT ---------------------------------//
/**
 * Contains parameters and SIMD instructions specific for certain score type.
 */
template<typename T> class SimdG {};

template<>
struct SimdG<char> {
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
struct SimdG<short> {
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
struct SimdG<int> {
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




template<class SIMD>
static int swimdSearchDatabaseGlobal_(unsigned char query[], int queryLength, 
                                      unsigned char** db, int dbLength, int dbSeqLengths[],
                                      int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                                      int scores[]) {

    static const typename SIMD::type LOWER_BOUND = std::numeric_limits<typename SIMD::type>::min();
    static const typename SIMD::type UPPER_BOUND = std::numeric_limits<typename SIMD::type>::max();
    // Used to represent -inf. Must be larger then lower bound to avoid overflow.
    static const typename SIMD::type LOWER_SCORE_BOUND = LOWER_BOUND + gapExt;

    // ----------------------- CHECK ARGUMENTS -------------------------- //
    // Check if Q, R or scoreMatrix have values too big for used score type
    if (gapOpen < LOWER_BOUND || UPPER_BOUND < gapOpen || gapExt < LOWER_BOUND || UPPER_BOUND < gapExt) {
        return 1;
    }
    if (!SIMD::satArthm) {
        // These extra limits are enforced so overflow could be detected more efficiently
        if (gapOpen <= LOWER_BOUND/2 || UPPER_BOUND/2 <= gapOpen || gapExt <= LOWER_BOUND/2 || UPPER_BOUND/2 <= gapExt) {
            return 1;
        }
    }

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

    // TODO: Check if inital values of H (-gapOpen - i * gapExt) cause overflow
    // ------------------------------------------------------------------ //


    // ------------------------ INITIALIZATION -------------------------- //
    for (int i = 0; i < dbLength; i++)
        scores[i] = -1;

    __m128i zeroes = SIMD::set1(0);
    __m128i lowerBounds = SIMD::set1(LOWER_BOUND);
    __m128i lowerScoreBounds = SIMD::set1(LOWER_SCORE_BOUND);

    int nextDbSeqIdx = 0; // index in db
    int currDbSeqsIdxs[SIMD::numSeqs]; // index in db
    unsigned char* currDbSeqsPos[SIMD::numSeqs]; // current element for each current database sequence
    int currDbSeqsLengths[SIMD::numSeqs];
    int shortestDbSeqLength = -1;  // length of shortest sequence among current database sequences
    int numEndedDbSeqs = 0; // Number of sequences that ended

    // Load initial sequences
    for (int i = 0; i < SIMD::numSeqs; i++)
        if (loadNextSequence(nextDbSeqIdx, dbLength, currDbSeqsIdxs[i], currDbSeqsPos[i],
                             currDbSeqsLengths[i], db, dbSeqLengths)) {
            // Update shortest sequence length if new sequence was loaded
            if (shortestDbSeqLength == -1 || currDbSeqsLengths[i] < shortestDbSeqLength)
                shortestDbSeqLength = currDbSeqsLengths[i];
        }

    // Q is gap open penalty, R is gap ext penalty.
    __m128i Q = SIMD::set1(gapOpen);
    __m128i R = SIMD::set1(gapExt);

    // Previous H column (array), previous E column (array), previous F, all signed short
    __m128i prevHs[queryLength];
    __m128i prevEs[queryLength];
    // Initialize all values
    for (int i = 0; i < queryLength; i++) {
        prevHs[i] = zeroes; // Or -gapOpen - i * gapExt if bounded
        prevEs[i] = lowerScoreBounds;
    }

    __m128i maxLastRowH = lowerBounds; // Keeps track of maximum H in last row
    // ------------------------------------------------------------------ //


    int columnsSinceLastSeqEnd = 0;
    // For each column
    while (numEndedDbSeqs < dbLength) {	
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
        uF = lowerScoreBounds;
        uH = zeroes;
        ulH = zeroes; // TODO: take care it is always zero for first row of first column

        __m128i minE, minF;
        minE = minF = SIMD::set1(UPPER_BOUND);
        __m128i maxH = lowerBounds; // Max H in this column
        __m128i H;

        // ----------------------- CORE LOOP (ONE COLUMN) ----------------------- //
        for (int r = 0; r < queryLength; r++) { // For each cell in column
            // Calculate E = max(lH-Q, lE-R)
            __m128i E = SIMD::max(SIMD::sub(prevHs[r], Q), SIMD::sub(prevEs[r], R)); // Should check E == LOWER_BOUND

            // Calculate F = max(uH-Q, uF-R)
            __m128i F = SIMD::max(SIMD::sub(uH, Q), SIMD::sub(uF, R)); // Should check F == LOWER_BOUND
            minF = SIMD::min(minF, F); // For overflow detection

            // Calculate H
    	    H = SIMD::max(F, E);
            __m128i ulH_P = SIMD::add(ulH, P[query[r]]); 
            H = SIMD::max(H, ulH_P); // Check if H == UPPER_BOUND

            maxH = SIMD::max(maxH, H); // update best score in column -> check if maxH == UPPER_BOUND
            
            // TODO: TREBA SMISLITI KAKO OVO SVE TEMPLEJTIZIRATI, ZASAD RADIM ZA OV

            // Set uF, uH, ulH
            uF = F;
            uH = H;
            ulH = prevHs[r];

            // Update prevHs, prevEs in advance for next column
            prevEs[r] = E;
            prevHs[r] = H;
        }
        // ---------------------------------------------------------------------- //

        // Find minE, which should be checked with minE == LOWER_BOUND for overflow
        for (int r = 0; r < queryLength; r++)
            minE = SIMD::min(minE, prevEs[r]);

        maxLastRowH = SIMD::max(maxLastRowH, H);

        columnsSinceLastSeqEnd++;
        typename SIMD::type* unpackedMaxH = (typename SIMD::type *)&maxH;
	
        // ------------------------ OVERFLOW DETECTION -------------------------- //
        if (!SIMD::satArthm) {
            /*           // This check is based on following assumptions: 
            //  - overflow wraps
            //  - Q, R and all scores from scoreMatrix are between LOWER_BOUND/2 and UPPER_BOUND/2 exclusive
            typename SIMD::type* unpackedOfTest = (typename SIMD::type *)&ofTest;
            for (int i = 0; i < SIMD::numSeqs; i++)
                if (currDbSeqsPos[i] != 0 && unpackedOfTest[i] <= LOWER_BOUND/2)
                return 1;*/
        } else {
            // There is overflow if minE == LOWER_BOUND or minF == LOWER_BOUND or maxH == UPPER_BOUND
            minE = SIMD::min(minE, minF);
            typename SIMD::type* unpackedMinEF = (typename SIMD::type *)&minE;
            for (int i = 0; i < SIMD::numSeqs; i++)
                if (currDbSeqsPos[i] != 0)
                    if (unpackedMinEF[i] == LOWER_BOUND || unpackedMaxH[i] == UPPER_BOUND) {
                        return SWIMD_ERR_OVERFLOW;
                    }
        }
        // ---------------------------------------------------------------------- //

        // --------------------- CHECK AND HANDLE SEQUENCE END ------------------ //
        if (shortestDbSeqLength == columnsSinceLastSeqEnd) { // If at least one sequence ended
            shortestDbSeqLength = -1;
            bool reset[SIMD::numSeqs]; // True if channel i should be reseted

            for (int i = 0; i < SIMD::numSeqs; i++) {
                if (currDbSeqsPos[i] != 0) { // If not null sequence
                    currDbSeqsLengths[i] -= columnsSinceLastSeqEnd;
                    if (currDbSeqsLengths[i] == 0) { // If sequence ended
                        numEndedDbSeqs++;
                        // Save best sequence score
                        __m128i bestScore = SIMD::max(maxH, maxLastRowH); // Maximum of last row and column
                        typename SIMD::type* unpackedBestScore = (typename SIMD::type *)&bestScore;
                        scores[currDbSeqsIdxs[i]] = unpackedBestScore[i];
                        // Load next sequence
                        loadNextSequence(nextDbSeqIdx, dbLength, currDbSeqsIdxs[i], currDbSeqsPos[i],
                                         currDbSeqsLengths[i], db, dbSeqLengths);
                        reset[i] = true;
                    } else {
                        reset[i] = false;
                            
                        if (currDbSeqsPos[i] != 0)
                            currDbSeqsPos[i]++; // If not new and not null, move for one element
                    }
                    // Update shortest sequence length if sequence is not null
                    if (currDbSeqsPos[i] != 0 && (shortestDbSeqLength == -1 || currDbSeqsLengths[i] < shortestDbSeqLength))
                        shortestDbSeqLength = currDbSeqsLengths[i];
                }
            }
            //------------ Reset prevEs, prevHs and maxLastRowH ------------//
            typename SIMD::type resetMask[SIMD::numSeqs] __attribute__((aligned(16)));
            __m128i resetMaskPacked;

            // Reset channels to zero
            for (int i = 0; i < SIMD::numSeqs; i++)
                resetMask[i] = reset[i] ? 0 : -1;
            resetMaskPacked = _mm_load_si128((__m128i const*)resetMask);
            for (int r = 0; r < queryLength; r++) {
                prevEs[r] = _mm_and_si128(prevEs[r], resetMaskPacked);
                prevHs[r] = _mm_and_si128(prevHs[r], resetMaskPacked);
            }
            maxLastRowH = _mm_and_si128(maxLastRowH, resetMaskPacked);

            // Add LOWER_SCORE_BOUND to channels
            for (int i = 0; i < SIMD::numSeqs; i++)
                resetMask[i] = reset[i] ? LOWER_SCORE_BOUND : 0;
            resetMaskPacked = _mm_load_si128((__m128i const*)resetMask);
            for (int r = 0; r < queryLength; r++) {
                prevEs[r] = SIMD::add(prevEs[r], resetMaskPacked);
            }

            // Add LOWER_BOUND to maxLastRow
            for (int i = 0; i < SIMD::numSeqs; i++)
                resetMask[i] = reset[i] ? LOWER_BOUND : 0;
            resetMaskPacked = _mm_load_si128((__m128i const*)resetMask);
            maxLastRowH = SIMD::add(maxLastRowH, resetMaskPacked);
            //-------------------------------------------------------//

            columnsSinceLastSeqEnd = 0;
        } else { // If no sequences ended
            // Move for one element in all sequences
            for (int i = 0; i < SIMD::numSeqs; i++)
                if (currDbSeqsPos[i] != 0)
                    currDbSeqsPos[i]++;
        }
        // ---------------------------------------------------------------------- //
    }

    return 0;
}

extern int swimdSearchDatabaseGlobal(unsigned char query[], int queryLength, 
                                     unsigned char** db, int dbLength, int dbSeqLengths[],
                                     int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                                     int scores[]) {
    #ifndef __SSE4_1__
    return SWIMD_ERR_NO_SIMD_SUPPORT;
    #endif

    int resultCode;
    printf("Using char\n");
    resultCode = swimdSearchDatabaseGlobal_< SimdG<char> >(query, queryLength, 
                                                          db, dbLength, dbSeqLengths, 
                                                          gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
    if (resultCode != 0) {
        printf("Using short\n");
	    resultCode = swimdSearchDatabaseGlobal_< SimdG<short> >(query, queryLength, 
                                                               db, dbLength, dbSeqLengths,
                                                               gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
	    if (resultCode != 0) {
            printf("Using int\n");
	        resultCode = swimdSearchDatabaseGlobal_< SimdG<int> >(query, queryLength, 
                                                                 db, dbLength, dbSeqLengths,
                                                                 gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
        }
    }

    return resultCode;
}
