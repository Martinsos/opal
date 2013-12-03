#ifndef SWIMD_H
#define SWIMD_H

/***********************************************************************************
 *  - adapts score type and reports overflow if score does not fit in int
 *  - calculating column by column (not 4 of them at once)
 *  - db sequences are not padded
 *  - using saturated arithmetic when possible
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
 *           - IMPORTANT! basic idea: return map instead of vector, use partial results when overflow happens
 *           - another idea: dont stop when overflow but go to next sequence, and report all which had overflow
 *           - third idea: do calculation like in basic idea but in blocks, so smaller precisions get more chance
 *************************************************************************************/

#ifdef __cplusplus 
extern "C" {
#endif

    // Error codes for databaseSearch
#define SWIMD_ERR_OVERFLOW 1

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
 * @param [out] scores Contains score for every database sequence (-1 if not calculated)
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
