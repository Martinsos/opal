#ifndef SWIMD_H
#define SWIMD_H

/***********************************************************************************
 *  - adapts score type and reports overflow if score does not fit in int
 *  - calculating column by column (not 4 of them at once)
 *  - db sequences are not padded
 *  - using saturation arithmetic when possible
 *  - using cunning way to detect overflow when not using saturation arithmetic
 *  - works for SSE4.1 and higher
 * 
 * TODO: - maybe add support for ending sequence earlier if it can not be better than best result
 *       - check if data is aligned! (But is it important for integers and shorts, or only for floats?)
 *       - HM? use unsigned values to get bigger range (can that be done?)
 *       - HM? use as less registers as possible! Investigate assembly code to determine how to do it.
 *       - maybe pad columns so length%4 == 0, that way I can check for sequence ending only on every 4th column, not every column
 *       - put consts where possible
 *       - do some serious profiling
 *       - does SSE overflow wrap?
 *       - Do genomes usually work with score matrices or just mismatch and match values?
 *       - think of some intelligent way on deciding which precision to use when
 *           - when overflow happens, do not calculate what has been calculated
 *           - keep track of N last scores, if they are low enough, switch to lower precision
 *           - another idea: do calculation like in basic idea but in blocks, so smaller precisions get more chance
 *           - another idea: first calculate all with byte, then those that overflowed with higher precision and so on
 *************************************************************************************/

#ifdef __cplusplus 
extern "C" {
#endif

// Error codes for databaseSearch
#define SWIMD_ERR_OVERFLOW 1 //!< Returned when score overflow happens.
#define SWIMD_ERR_NO_SIMD_SUPPORT 2 //!< Returned if available SIMD is not SSE4.1 or higher.

/**
 * Compares query sequence with each database sequence and returns similarity scores.
 * Sequences are not represented as arrays of letters, but as arrays of indices of letters in alphabet.
 * For example, if alphabet is {A,C,T,G} and sequence is ACCTCAG it will be represented as 0112103.
 * Opening of gap is penalized with gapOpen, while gap extension is penalized with gapExt.
 * gapOpen, gapExt and scores from scoreMatrix must be in (INT_MIN/2, INT_MAX/2).
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
 * @return 0 if all okay, error code otherwise.
 */
int swimdSearchDatabase(unsigned char query[], int queryLength, 
                        unsigned char** db, int dbLength, int dbSeqLengths[],
                        int gapOpen, int gapExt, int* scoreMatrix, int alphabetLength,
                        int scores[]);

#ifdef __cplusplus 
}
#endif

#endif /* SWIMD_H */
