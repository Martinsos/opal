#ifndef SWIMD_H
#define SWIMD_H

/***********************************************************************************
 *  - adapts score type and reports overflow if score does not fit in int
 *  - calculating column by column (not 4 of them at once)
 *  - db sequences are not padded
 *  - using saturation arithmetic when possible
 *  - works for SSE4.1 and higher
 *************************************************************************************/

#ifdef __cplusplus 
extern "C" {
#endif

// Error codes
#define SWIMD_ERR_OVERFLOW 1 //!< Returned when score overflow happens.
#define SWIMD_ERR_NO_SIMD_SUPPORT 2 //!< Returned if available SIMD is not SSE4.1 or higher.
#define SWIMD_ERR_INVALID_MODE 3 //!< Returned when given mode is invalid.
    
// Modes
#define SWIMD_MODE_NW 0
#define SWIMD_MODE_HW 1
#define SWIMD_MODE_OV 2
#define SWIMD_MODE_SW 3
    
 /**
 * Compares query sequence with each database sequence and returns similarity scores.
 * Uses one of following alignment algorithms in combination with afine gaps(GOTOH):
 *   SW, NW, HW, OV.
 * Sequences are not represented as arrays of letters, but as arrays of indices of letters in alphabet.
 * For example, if alphabet is {A,C,T,G} and sequence is ACCTCAG it will be represented as 0112103.
 * Opening of gap is penalized with gapOpen, while gap extension is penalized with gapExt.
 * gapOpen, gapExt and scores from scoreMatrix must be in (INT_MIN/2, INT_MAX/2).
 * Detects overflow only for SW!
 * @param [in] query Query sequence.
 * @param [in] queryLength Length of query sequence.
 * @param [in] db Array of database sequences (each sequence is also an array).
 * @param [in] dbLength Number of database sequences.
 * @param [in] dbSeqLengths Array of lengths of database sequences.
 * @param [in] gapOpen
 * @param [in] gapExt
 * @param [in] scoreMatrix Matrix of dimensions (alphabetLength, alphabetLength).
 * @param [in] alphabetLength
 * @param [out] scores Score for every database sequence is stored here (-1 if not calculated).
 * @param [in] mode Mode of alignment, different mode means different algorithm.
 *                  Valid modes: SWIMD_MODE_NW, SWIMD_MODE_HW, SWIMD_MODE_OV, SWIMD_MODE_SW.
 * @return 0 if all okay, error code otherwise.
 */
int swimdSearchDatabase(unsigned char query[], int queryLength, 
                        unsigned char** db, int dbLength, int dbSeqLengths[],
                        int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                        int scores[], const int mode);
    
#ifdef __cplusplus 
}
#endif

#endif /* SWIMD_H */
