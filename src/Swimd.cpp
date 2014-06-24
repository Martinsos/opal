#include <cstdio>
#include <limits>

extern "C" {
#include <immintrin.h> // AVX2 and lower
}

#include "Swimd.h"


// I define aliases for SSE intrinsics, so they can be used in code not depending on SSE generation.
// If available, AVX2 is used because it has two times bigger register, thus everything is two times faster.
#ifdef __AVX2__

const int SIMD_REG_SIZE = 256; //!< number of bits in register
typedef __m256i __mxxxi; //!< represents register containing integers
#define _mmxxx_load_si  _mm256_load_si256
#define _mmxxx_store_si _mm256_store_si256
#define _mmxxx_and_si   _mm256_and_si256

#define _mmxxx_adds_epi8 _mm256_adds_epi8
#define _mmxxx_subs_epi8 _mm256_subs_epi8
#define _mmxxx_min_epu8  _mm256_min_epu8
#define _mmxxx_min_epi8  _mm256_min_epi8
#define _mmxxx_max_epu8  _mm256_max_epu8
#define _mmxxx_max_epi8  _mm256_max_epi8
#define _mmxxx_set1_epi8 _mm256_set1_epi8

#define _mmxxx_adds_epi16 _mm256_adds_epi16
#define _mmxxx_subs_epi16 _mm256_subs_epi16
#define _mmxxx_min_epi16  _mm256_min_epi16
#define _mmxxx_max_epi16  _mm256_max_epi16
#define _mmxxx_set1_epi16 _mm256_set1_epi16

#define _mmxxx_add_epi32 _mm256_add_epi32
#define _mmxxx_sub_epi32 _mm256_sub_epi32
#define _mmxxx_min_epi32  _mm256_min_epi32
#define _mmxxx_max_epi32  _mm256_max_epi32
#define _mmxxx_set1_epi32 _mm256_set1_epi32

#else // SSE4.1

const int SIMD_REG_SIZE = 128;
typedef __m128i __mxxxi;
#define _mmxxx_load_si  _mm_load_si128
#define _mmxxx_store_si _mm_store_si128
#define _mmxxx_and_si   _mm_and_si128

#define _mmxxx_adds_epi8 _mm_adds_epi8
#define _mmxxx_subs_epi8 _mm_subs_epi8
#define _mmxxx_min_epu8  _mm_min_epu8
#define _mmxxx_min_epi8  _mm_min_epi8
#define _mmxxx_max_epu8  _mm_max_epu8
#define _mmxxx_max_epi8  _mm_max_epi8
#define _mmxxx_set1_epi8 _mm_set1_epi8

#define _mmxxx_adds_epi16 _mm_adds_epi16
#define _mmxxx_subs_epi16 _mm_subs_epi16
#define _mmxxx_min_epi16  _mm_min_epi16
#define _mmxxx_max_epi16  _mm_max_epi16
#define _mmxxx_set1_epi16 _mm_set1_epi16

#define _mmxxx_add_epi32 _mm_add_epi32
#define _mmxxx_sub_epi32 _mm_sub_epi32
#define _mmxxx_min_epi32  _mm_min_epi32
#define _mmxxx_max_epi32  _mm_max_epi32
#define _mmxxx_set1_epi32 _mm_set1_epi32

#endif


//------------------------------------ SIMD PARAMETERS ---------------------------------//
/**
 * Contains parameters and SIMD instructions specific for certain score type.
 */
template<typename T> class SimdSW {};

template<>
struct SimdSW<char> {
    typedef char type; //!< Type that will be used for score
    static const int numSeqs = SIMD_REG_SIZE / (8 * sizeof(char)); //!< Number of sequences that can be done in parallel.
    static const bool satArthm = true; //!< True if saturation arithmetic is used, false otherwise.
    static const bool negRange = true; //!< True if it uses negative range for score representation, goes with saturation
    static inline __mxxxi add(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_adds_epi8(a, b); }
    static inline __mxxxi sub(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_subs_epi8(a, b); }
    static inline __mxxxi min(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_min_epu8(a, b); }
    static inline __mxxxi max(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_max_epu8(a, b); }
    static inline __mxxxi set1(int a) { return _mmxxx_set1_epi8(a); }
};

template<>
struct SimdSW<short> {
    typedef short type;
    static const int numSeqs = SIMD_REG_SIZE / (8 * sizeof(short));
    static const bool satArthm = true;
    static const bool negRange = false;
    static inline __mxxxi add(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_adds_epi16(a, b); }
    static inline __mxxxi sub(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_subs_epi16(a, b); }
    static inline __mxxxi min(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_min_epi16(a, b); }
    static inline __mxxxi max(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_max_epi16(a, b); }
    static inline __mxxxi set1(int a) { return _mmxxx_set1_epi16(a); }
};

template<>
struct SimdSW<int> {
    typedef int type;
    static const int numSeqs = SIMD_REG_SIZE / (8 * sizeof(int));
    static const bool satArthm = false;
    static const bool negRange = false;
    static inline __mxxxi add(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_add_epi32(a, b); }
    static inline __mxxxi sub(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_sub_epi32(a, b); }
    static inline __mxxxi min(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_min_epi32(a, b); }
    static inline __mxxxi max(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_max_epi32(a, b); }
    static inline __mxxxi set1(int a) { return _mmxxx_set1_epi32(a); }
};
//--------------------------------------------------------------------------------------//


static bool loadNextSequence(int &nextDbSeqIdx, int dbLength, int &currDbSeqIdx, unsigned char * &currDbSeqPos, 
                             int &currDbSeqLength, unsigned char ** db, int dbSeqLengths[], bool calculated[],
                             int &numEndedDbSeqs);


// For debugging
template<class SIMD>
void print_mmxxxi(__mxxxi mm) {
    typename SIMD::type unpacked[SIMD::numSeqs];
    _mmxxx_store_si((__mxxxi*)unpacked, mm);
    for (int i = 0; i < SIMD::numSeqs; i++)
        printf("%d ", unpacked[i]);
}

template<class SIMD>
static int searchDatabaseSW_(unsigned char query[], int queryLength, 
                             unsigned char** db, int dbLength, int dbSeqLengths[],
                             int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                             int scores[], bool calculated[]) {

    const typename SIMD::type LOWER_BOUND = std::numeric_limits<typename SIMD::type>::min();
    const typename SIMD::type UPPER_BOUND = std::numeric_limits<typename SIMD::type>::max();

    // ----------------------- CHECK ARGUMENTS -------------------------- //
    // Check if Q, R or scoreMatrix have values too big for used score type
    if (gapOpen < LOWER_BOUND || UPPER_BOUND < gapOpen || gapExt < LOWER_BOUND || UPPER_BOUND < gapExt) {
        return 1;
    }
    if (!SIMD::satArthm) {
        // These extra limits are enforced so overflow could be detected more efficiently
        if (gapOpen <= LOWER_BOUND/2 || UPPER_BOUND/2 <= gapOpen || gapExt <= LOWER_BOUND/2 || UPPER_BOUND/2 <= gapExt) {
            return SWIMD_ERR_OVERFLOW;
        }
    }

    for (int r = 0; r < alphabetLength; r++)
        for (int c = 0; c < alphabetLength; c++) {
            int score = scoreMatrix[r * alphabetLength + c];
            if (score < LOWER_BOUND || UPPER_BOUND < score) {
                return SWIMD_ERR_OVERFLOW;
            }
            if (!SIMD::satArthm) {
                if (score <= LOWER_BOUND/2 || UPPER_BOUND/2 <= score)
                    return SWIMD_ERR_OVERFLOW;
            }
        }
    // ------------------------------------------------------------------ //


    // ------------------------ INITIALIZATION -------------------------- //
    __mxxxi zeroes = SIMD::set1(0);
    __mxxxi scoreZeroes; // 0 normally, but lower bound if using negative range
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
                             currDbSeqsLengths[i], db, dbSeqLengths, calculated, numEndedDbSeqs)) {
            // Update shortest sequence length if new sequence was loaded
            if (shortestDbSeqLength == -1 || currDbSeqsLengths[i] < shortestDbSeqLength)
                shortestDbSeqLength = currDbSeqsLengths[i];
        }

    // Q is gap open penalty, R is gap ext penalty.
    __mxxxi Q = SIMD::set1(gapOpen);
    __mxxxi R = SIMD::set1(gapExt);

    // Previous H column (array), previous E column (array), previous F, all signed short
    __mxxxi prevHs[queryLength];
    __mxxxi prevEs[queryLength];
    // Initialize all values to 0
    for (int i = 0; i < queryLength; i++) {
        prevHs[i] = prevEs[i] = scoreZeroes;
    }

    __mxxxi maxH = scoreZeroes;  // Best score in sequence
    // ------------------------------------------------------------------ //


    int columnsSinceLastSeqEnd = 0;
    // For each column
    while (numEndedDbSeqs < dbLength) {
        // -------------------- CALCULATE QUERY PROFILE ------------------------- //
        // TODO: Rognes uses pshufb here, I don't know how/why?
        __mxxxi P[alphabetLength];
        typename SIMD::type profileRow[SIMD::numSeqs] __attribute__((aligned(16)));
        for (unsigned char letter = 0; letter < alphabetLength; letter++) {
            int* scoreMatrixRow = scoreMatrix + letter*alphabetLength;
            for (int i = 0; i < SIMD::numSeqs; i++) {
                unsigned char* dbSeqPos = currDbSeqsPos[i];
                if (dbSeqPos != 0)
                    profileRow[i] = (typename SIMD::type)scoreMatrixRow[*dbSeqPos];
            }
            P[letter] = _mmxxx_load_si((__mxxxi const*)profileRow);
        }
        // ---------------------------------------------------------------------- //
        
        // Previous cells: u - up, l - left, ul - up left
        __mxxxi uF, uH, ulH; 
        uF = uH = ulH = scoreZeroes; // F[-1, c] = H[-1, c] = H[-1, c-1] = 0

        __mxxxi ofTest = scoreZeroes; // Used for detecting the overflow when not using saturated ar

        // ----------------------- CORE LOOP (ONE COLUMN) ----------------------- //
        for (int r = 0; r < queryLength; r++) { // For each cell in column
            // Calculate E = max(lH-Q, lE-R)
            __mxxxi E = SIMD::max(SIMD::sub(prevHs[r], Q), SIMD::sub(prevEs[r], R));

            // Calculate F = max(uH-Q, uF-R)
            __mxxxi F = SIMD::max(SIMD::sub(uH, Q), SIMD::sub(uF, R));

            // Calculate H
            __mxxxi H = SIMD::max(F, E);
            if (!SIMD::negRange) // If not using negative range, then H could be negative at this moment so we need this
                H = SIMD::max(H, zeroes);
            __mxxxi ulH_P = SIMD::add(ulH, P[query[r]]); // If using negative range: if ulH_P >= 0 then we have overflow

            H = SIMD::max(H, ulH_P); // If using negative range: H will always be negative, even if ulH_P overflowed

            // Save data needed for overflow detection. Not more then one condition will fire
            if (SIMD::negRange)
                ofTest = _mmxxx_and_si(ofTest, ulH_P);
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

        typename SIMD::type unpackedMaxH[SIMD::numSeqs];
        _mmxxx_store_si((__mxxxi*)unpackedMaxH, maxH);

        // ------------------------ OVERFLOW DETECTION -------------------------- //
        if (!SIMD::satArthm) {
            // This check is based on following assumptions: 
            //  - overflow wraps
            //  - Q, R and all scores from scoreMatrix are between LOWER_BOUND/2 and UPPER_BOUND/2 exclusive
            typename SIMD::type unpackedOfTest[SIMD::numSeqs];
            _mmxxx_store_si((__mxxxi*)unpackedOfTest, ofTest);
            for (int i = 0; i < SIMD::numSeqs; i++)
                if (currDbSeqsPos[i] != 0 && unpackedOfTest[i] <= LOWER_BOUND/2)
                    return SWIMD_ERR_OVERFLOW;
        } else {
            if (SIMD::negRange) {
                // Since I use saturation, I check if minUlH_P was non negative
                typename SIMD::type unpackedOfTest[SIMD::numSeqs];
                _mmxxx_store_si((__mxxxi*)unpackedOfTest, ofTest);
                for (int i = 0; i < SIMD::numSeqs; i++)
                    if (currDbSeqsPos[i] != 0 && unpackedOfTest[i] >= 0)
                        return SWIMD_ERR_OVERFLOW;
            } else {
                // I check if upper bound is reached
                for (int i = 0; i < SIMD::numSeqs; i++)
                    if (currDbSeqsPos[i] != 0 && unpackedMaxH[i] == UPPER_BOUND) {
                        return SWIMD_ERR_OVERFLOW;
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
                        calculated[currDbSeqsIdxs[i]] = true;
                        // Load next sequence
                        loadNextSequence(nextDbSeqIdx, dbLength, currDbSeqsIdxs[i], currDbSeqsPos[i],
                                         currDbSeqsLengths[i], db, dbSeqLengths, calculated, numEndedDbSeqs);
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
            __mxxxi resetMaskPacked = _mmxxx_load_si((__mxxxi const*)resetMask);
            if (SIMD::negRange) {
                for (int i = 0; i < queryLength; i++)
                    prevEs[i] = SIMD::add(prevEs[i], resetMaskPacked);
                for (int i = 0; i < queryLength; i++)
                    prevHs[i] = SIMD::add(prevHs[i], resetMaskPacked);
                maxH = SIMD::add(maxH, resetMaskPacked);
            } else {
                for (int i = 0; i < queryLength; i++)
                    prevEs[i] = _mmxxx_and_si(prevEs[i], resetMaskPacked);
                for (int i = 0; i < queryLength; i++)
                    prevHs[i] = _mmxxx_and_si(prevHs[i], resetMaskPacked);
                maxH = _mmxxx_and_si(maxH, resetMaskPacked);
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
                                    int &currDbSeqLength, unsigned char** db, int dbSeqLengths[], bool calculated[],
                                    int &numEndedDbSeqs) {
    while (nextDbSeqIdx < dbLength && calculated[nextDbSeqIdx]) {
        nextDbSeqIdx++;
        numEndedDbSeqs++;
    }
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

extern int searchDatabaseSW(unsigned char query[], int queryLength, 
                            unsigned char** db, int dbLength, int dbSeqLengths[],
                            int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                            int scores[]) {
    int resultCode = 0;
    const int chunkSize = 1024;
    bool* calculated = new bool[chunkSize];
    for (int startIdx = 0; startIdx < dbLength; startIdx += chunkSize) {
        unsigned char** db_ = db + startIdx;
        int* dbSeqLengths_ = dbSeqLengths + startIdx;
        int* scores_ = scores + startIdx;
        int dbLength_ = startIdx + chunkSize >= dbLength ? dbLength - startIdx : chunkSize;
        for (int i = 0; i < dbLength_; i++)
            calculated[i] = false;
        resultCode = searchDatabaseSW_< SimdSW<char> >(query, queryLength, 
                                                       db_, dbLength_, dbSeqLengths_, 
                                                       gapOpen, gapExt, scoreMatrix, alphabetLength, scores_,
                                                       calculated);
        if (resultCode != 0) {
            resultCode = searchDatabaseSW_< SimdSW<short> >(query, queryLength,
                                                            db_, dbLength_, dbSeqLengths_,
                                                            gapOpen, gapExt, scoreMatrix, alphabetLength, scores_,
                                                            calculated);
            if (resultCode != 0) {
                resultCode = searchDatabaseSW_< SimdSW<int> >(query, queryLength,
                                                              db_, dbLength_, dbSeqLengths_,
                                                              gapOpen, gapExt, scoreMatrix, alphabetLength, scores_,
                                                              calculated);
                if (resultCode != 0)
                    break;
            }
        }
    }

    delete[] calculated;
    return resultCode;
}









//------------------------------------ SIMD PARAMETERS ---------------------------------//
/**
 * Contains parameters and SIMD instructions specific for certain score type.
 */
template<typename T> class Simd {};

template<>
struct Simd<char> {
    typedef char type; //!< Type that will be used for score
    static const int numSeqs = SIMD_REG_SIZE / (8 * sizeof(char)); //!< Number of sequences that can be done in parallel.
    static const bool satArthm = true; //!< True if saturation arithmetic is used, false otherwise.
    static inline __mxxxi add(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_adds_epi8(a, b); }
    static inline __mxxxi sub(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_subs_epi8(a, b); }
    static inline __mxxxi min(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_min_epi8(a, b); }
    static inline __mxxxi max(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_max_epi8(a, b); }
    static inline __mxxxi set1(int a) { return _mmxxx_set1_epi8(a); }
};

template<>
struct Simd<short> {
    typedef short type;
    static const int numSeqs = SIMD_REG_SIZE / (8 * sizeof(short));
    static const bool satArthm = true;
    static inline __mxxxi add(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_adds_epi16(a, b); }
    static inline __mxxxi sub(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_subs_epi16(a, b); }
    static inline __mxxxi min(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_min_epi16(a, b); }
    static inline __mxxxi max(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_max_epi16(a, b); }
    static inline __mxxxi set1(int a) { return _mmxxx_set1_epi16(a); }
};

template<>
struct Simd<int> {
    typedef int type;
    static const int numSeqs = SIMD_REG_SIZE / (8 * sizeof(int));
    static const bool satArthm = false;
    static inline __mxxxi add(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_add_epi32(a, b); }
    static inline __mxxxi sub(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_sub_epi32(a, b); }
    static inline __mxxxi min(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_min_epi32(a, b); }
    static inline __mxxxi max(const __mxxxi& a, const __mxxxi& b) { return _mmxxx_max_epi32(a, b); }
    static inline __mxxxi set1(int a) { return _mmxxx_set1_epi32(a); }
};
//--------------------------------------------------------------------------------------//




template<class SIMD, int MODE>
static int searchDatabase_(unsigned char query[], int queryLength, 
                           unsigned char** db, int dbLength, int dbSeqLengths[],
                           int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                           int scores[], bool calculated[]) {

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

    // TODO: If not saturated arithmetic check if inital values of H (-gapOpen - i * gapExt) cause overflow
    // ------------------------------------------------------------------ //


    // ------------------------ INITIALIZATION -------------------------- //
    const __mxxxi ZERO_SIMD = SIMD::set1(0);
    const __mxxxi LOWER_BOUND_SIMD = SIMD::set1(LOWER_BOUND);
    const __mxxxi LOWER_SCORE_BOUND_SIMD = SIMD::set1(LOWER_SCORE_BOUND);
    
    int nextDbSeqIdx = 0; // index in db
    int currDbSeqsIdxs[SIMD::numSeqs]; // index in db
    unsigned char* currDbSeqsPos[SIMD::numSeqs]; // current element for each current database sequence
    int currDbSeqsLengths[SIMD::numSeqs];
    bool justLoaded[SIMD::numSeqs] = {0}; // True if sequence was just loaded into channel
    bool seqJustLoaded = false; // True if at least one sequence was just loaded into channel
    int shortestDbSeqLength = -1;  // length of shortest sequence among current database sequences
    int numEndedDbSeqs = 0; // Number of sequences that ended

    // Load initial sequences
    for (int i = 0; i < SIMD::numSeqs; i++)
        if (loadNextSequence(nextDbSeqIdx, dbLength, currDbSeqsIdxs[i], currDbSeqsPos[i],
                             currDbSeqsLengths[i], db, dbSeqLengths, calculated, numEndedDbSeqs)) {
            justLoaded[i] = seqJustLoaded = true;
            // Update shortest sequence length if new sequence was loaded
            if (shortestDbSeqLength == -1 || currDbSeqsLengths[i] < shortestDbSeqLength)
                shortestDbSeqLength = currDbSeqsLengths[i];
        }

    // Q is gap open penalty, R is gap ext penalty.
    const __mxxxi Q = SIMD::set1(gapOpen);
    const __mxxxi R = SIMD::set1(gapExt);

    // Previous H column (array), previous E column (array), previous F, all signed short
    __mxxxi prevHs[queryLength];
    __mxxxi prevEs[queryLength];
    // Initialize all values
    for (int r = 0; r < queryLength; r++) {
        if (MODE == SWIMD_MODE_OV)
            prevHs[r] = ZERO_SIMD;
        else { // - Q - r * R
            if (r == 0)
                prevHs[0] = SIMD::sub(ZERO_SIMD, Q);
            else
                prevHs[r] = SIMD::sub(prevHs[r-1], R);
        }
        
        prevEs[r] = LOWER_SCORE_BOUND_SIMD;
    }

    // u - up, ul - up left
    __mxxxi uH, ulH;
    if (MODE == SWIMD_MODE_NW) {
        ulH = ZERO_SIMD;
        uH = SIMD::sub(R, Q); // -Q + R
    }

    __mxxxi maxLastRowH = LOWER_BOUND_SIMD; // Keeps track of maximum H in last row
    // ------------------------------------------------------------------ //


    int columnsSinceLastSeqEnd = 0;
    // For each column
    while (numEndedDbSeqs < dbLength) {
        // -------------------- CALCULATE QUERY PROFILE ------------------------- //
        // TODO: Rognes uses pshufb here, I don't know how/why?
        __mxxxi P[alphabetLength];
        typename SIMD::type profileRow[SIMD::numSeqs] __attribute__((aligned(16)));
        for (unsigned char letter = 0; letter < alphabetLength; letter++) {
            int* scoreMatrixRow = scoreMatrix + letter*alphabetLength;
            for (int i = 0; i < SIMD::numSeqs; i++) {
                unsigned char* dbSeqPos = currDbSeqsPos[i];
                if (dbSeqPos != 0)
                    profileRow[i] = (typename SIMD::type)scoreMatrixRow[*dbSeqPos];
            }
            P[letter] = _mmxxx_load_si((__mxxxi const*)profileRow);
        }
        // ---------------------------------------------------------------------- //

        // u - up
        __mxxxi uF = LOWER_SCORE_BOUND_SIMD;

        // Database sequence has fixed start and end only in NW
        if (MODE == SWIMD_MODE_NW) {
            if (seqJustLoaded) {
                typename SIMD::type resetMask[SIMD::numSeqs] __attribute__((aligned(16)));
                for (int i = 0; i < SIMD::numSeqs; i++) 
                    resetMask[i] = justLoaded[i] ?  0 : -1;
                const __mxxxi resetMaskPacked = _mmxxx_load_si((__mxxxi const*)resetMask);
                ulH = _mmxxx_and_si(uH, resetMaskPacked);
            } else {
                ulH = uH;
            }
            
            uH = SIMD::sub(uH, R); // uH is -Q - c*R
            // NOTE: Setup of ulH and uH for first column is done when sequence is loaded.
        } else {
            uH = ulH = ZERO_SIMD;
        }

        __mxxxi minE, minF;
        minE = minF = SIMD::set1(UPPER_BOUND);
        __mxxxi maxH = LOWER_BOUND_SIMD; // Max H in this column
        __mxxxi H;

        __mxxxi firstRow_uH, firstRow_ulH; // Values of uH and ulH from first row of column

        if (MODE == SWIMD_MODE_NW) {
            firstRow_uH = uH;
            firstRow_ulH = ulH;
        }
        
        // ----------------------- CORE LOOP (ONE COLUMN) ----------------------- //
        for (int r = 0; r < queryLength; r++) { // For each cell in column
            // Calculate E = max(lH-Q, lE-R)
            __mxxxi E = SIMD::max(SIMD::sub(prevHs[r], Q), SIMD::sub(prevEs[r], R)); // E could overflow

            // Calculate F = max(uH-Q, uF-R)
            __mxxxi F = SIMD::max(SIMD::sub(uH, Q), SIMD::sub(uF, R)); // F could overflow
            minF = SIMD::min(minF, F); // For overflow detection

            // Calculate H
            H = SIMD::max(F, E);
            __mxxxi ulH_P = SIMD::add(ulH, P[query[r]]); 
            H = SIMD::max(H, ulH_P); // H could overflow

            maxH = SIMD::max(maxH, H); // update best score in column

            // Set uF, uH, ulH
            uF = F;
            uH = H;
            ulH = prevHs[r];

            // Update prevHs, prevEs in advance for next column
            prevEs[r] = E;
            prevHs[r] = H;
        }
        // ---------------------------------------------------------------------- //

        maxLastRowH = SIMD::max(maxLastRowH, H);

        if (MODE == SWIMD_MODE_NW) {
            uH = firstRow_uH;
            ulH = firstRow_ulH;
        }

        // Find minE, which should be checked with minE == LOWER_BOUND for overflow
        for (int r = 0; r < queryLength; r++)
            minE = SIMD::min(minE, prevEs[r]);

        columnsSinceLastSeqEnd++;
        
        typename SIMD::type unpackedMaxH[SIMD::numSeqs];
        _mmxxx_store_si((__mxxxi*)unpackedMaxH, maxH);

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
            __mxxxi minEF = SIMD::min(minE, minF);
            typename SIMD::type unpackedMinEF[SIMD::numSeqs];
            _mmxxx_store_si((__mxxxi*)unpackedMinEF, minEF);
            for (int i = 0; i < SIMD::numSeqs; i++)
                if (currDbSeqsPos[i] != 0)
                    if (unpackedMinEF[i] == LOWER_BOUND || unpackedMaxH[i] == UPPER_BOUND) {
                        return SWIMD_ERR_OVERFLOW;
                    }
        }
        // ---------------------------------------------------------------------- //

        seqJustLoaded = false;
        // --------------------- CHECK AND HANDLE SEQUENCE END ------------------ //        
        if (shortestDbSeqLength == columnsSinceLastSeqEnd) { // If at least one sequence ended
            shortestDbSeqLength = -1;

            for (int i = 0; i < SIMD::numSeqs; i++) {
                if (currDbSeqsPos[i] != 0) { // If not null sequence
                    justLoaded[i] = false;
                    currDbSeqsLengths[i] -= columnsSinceLastSeqEnd;
                    
                    // Calculate best scores
                    __mxxxi bestScore;
                    if (MODE == SWIMD_MODE_OV)
                        bestScore = SIMD::max(maxH, maxLastRowH); // Maximum of last row and column
                    if (MODE == SWIMD_MODE_HW)
                        bestScore = maxLastRowH;
                    if (MODE == SWIMD_MODE_NW)
                        bestScore = H;
                    typename SIMD::type unpackedBestScore[SIMD::numSeqs];
                    _mmxxx_store_si((__mxxxi*)unpackedBestScore, bestScore);

                    if (currDbSeqsLengths[i] == 0) { // If sequence ended
                        numEndedDbSeqs++;
                        // Save best score
                        scores[currDbSeqsIdxs[i]] = unpackedBestScore[i];
                        calculated[currDbSeqsIdxs[i]] = true;
                        // Load next sequence
                        if (loadNextSequence(nextDbSeqIdx, dbLength, currDbSeqsIdxs[i], currDbSeqsPos[i],
                                             currDbSeqsLengths[i], db, dbSeqLengths, calculated, numEndedDbSeqs)) {
                            justLoaded[i] = seqJustLoaded = true;
                        }
                    } else {
                        if (currDbSeqsPos[i] != 0)
                            currDbSeqsPos[i]++; // If not new and not null, move for one element
                    }
                    // Update shortest sequence length if sequence is not null
                    if (currDbSeqsPos[i] != 0 && (shortestDbSeqLength == -1 || currDbSeqsLengths[i] < shortestDbSeqLength))
                        shortestDbSeqLength = currDbSeqsLengths[i];
                }
            }
            //------------ Reset prevEs, prevHs, maxLastRowH(, ulH and uH) ------------//
            typename SIMD::type resetMask[SIMD::numSeqs] __attribute__((aligned(16)));
            typename SIMD::type setMask[SIMD::numSeqs] __attribute__((aligned(16))); // inverse of resetMask
            for (int i = 0; i < SIMD::numSeqs; i++) {
                resetMask[i] = justLoaded[i] ?  0 : -1;
                setMask[i]   = justLoaded[i] ? -1 :  0;
            }
            const __mxxxi resetMaskPacked = _mmxxx_load_si((__mxxxi const*)resetMask);
            const __mxxxi setMaskPacked = _mmxxx_load_si((__mxxxi const*)setMask);

            // Set prevEs ended channels to LOWER_SCORE_BOUND
            const __mxxxi maskedLowerScoreBoundSimd = _mmxxx_and_si(setMaskPacked, LOWER_SCORE_BOUND_SIMD);
            for (int r = 0; r < queryLength; r++) {
                prevEs[r] = _mmxxx_and_si(prevEs[r], resetMaskPacked);
                prevEs[r] = SIMD::add(prevEs[r], maskedLowerScoreBoundSimd);
            }

            // Set prevHs
            for (int r = 0; r < queryLength; r++) {
                prevHs[r] = _mmxxx_and_si(prevHs[r], resetMaskPacked);
                if (MODE != SWIMD_MODE_OV) {
                    if (r == 0) {
                        prevHs[0] = SIMD::sub(prevHs[0], _mmxxx_and_si(setMaskPacked, Q));
                    } else {
                        prevHs[r] = SIMD::add(prevHs[r], _mmxxx_and_si(setMaskPacked, SIMD::sub(prevHs[r-1], R)));
                    }
                }
            }

            // Set ulH and uH if NW
            if (MODE == SWIMD_MODE_NW) {
                ulH = _mmxxx_and_si(ulH, resetMaskPacked); // to 0
                // Set uH channels to -Q + R
                uH = _mmxxx_and_si(uH, resetMaskPacked);
                uH = SIMD::add(uH, _mmxxx_and_si(setMaskPacked, SIMD::sub(R, Q)));
            }

            // Set maxLastRow ended channels to LOWER_BOUND
            maxLastRowH = _mmxxx_and_si(maxLastRowH, resetMaskPacked);
            maxLastRowH = SIMD::add(maxLastRowH, _mmxxx_and_si(setMaskPacked, LOWER_BOUND_SIMD));
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

template <int MODE>
static int searchDatabase(unsigned char query[], int queryLength, 
                          unsigned char** db, int dbLength, int dbSeqLengths[],
                          int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                          int scores[]) {
    int resultCode = 0;
    const int chunkSize = 1024;
    bool* calculated = new bool[chunkSize];
    for (int startIdx = 0; startIdx < dbLength; startIdx += chunkSize) {
        unsigned char** db_ = db + startIdx;
        int* dbSeqLengths_ = dbSeqLengths + startIdx;
        int* scores_ = scores + startIdx;
        int dbLength_ = startIdx + chunkSize >= dbLength ? dbLength - startIdx : chunkSize;
        for (int i = 0; i < dbLength_; i++)
            calculated[i] = false;
        resultCode = searchDatabase_< Simd<char>, MODE >
            (query, queryLength, db_, dbLength_, dbSeqLengths_, 
             gapOpen, gapExt, scoreMatrix, alphabetLength, scores_, calculated);
        if (resultCode != 0) {
            resultCode = searchDatabase_< Simd<short>, MODE >
                (query, queryLength, db_, dbLength_, dbSeqLengths_,
                 gapOpen, gapExt, scoreMatrix, alphabetLength, scores_, calculated);
            if (resultCode != 0) {
                resultCode = searchDatabase_< Simd<int>, MODE >
                    (query, queryLength, db_, dbLength_, dbSeqLengths_,
                     gapOpen, gapExt, scoreMatrix, alphabetLength, scores_, calculated);
                if (resultCode != 0)
                    break;
            }
        }
    }
    delete[] calculated;
    return resultCode;
}


extern int swimdSearchDatabase(unsigned char query[], int queryLength, 
                               unsigned char** db, int dbLength, int dbSeqLengths[],
                               int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                               int scores[], const int mode) {    
#if !defined(__SSE4_1__) && !defined(__AVX2__)
    return SWIMD_ERR_NO_SIMD_SUPPORT;
#else
    if (mode == SWIMD_MODE_NW) {
        return searchDatabase<SWIMD_MODE_NW>
            (query, queryLength, db, dbLength, dbSeqLengths, 
             gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
    } else if (mode == SWIMD_MODE_HW) {
        return searchDatabase<SWIMD_MODE_HW>
            (query, queryLength, db, dbLength, dbSeqLengths, 
             gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
    } else if (mode == SWIMD_MODE_OV) {
        return searchDatabase<SWIMD_MODE_OV>
            (query, queryLength, db, dbLength, dbSeqLengths, 
             gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
    } else if (mode == SWIMD_MODE_SW) {
        return searchDatabaseSW(query, queryLength, db, dbLength, dbSeqLengths, 
                                gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
    }
    return SWIMD_ERR_INVALID_MODE;
#endif
}
