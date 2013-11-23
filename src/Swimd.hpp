#ifndef SWIMD_HPP
#define SWIMD_HPP

#include <vector>
#include <climits>

using namespace std;

typedef unsigned char  Byte; // TODO: name this uchar?

#define OVERFLOW_EXC 1

// Macros for when using short
#define NUM_SEQS 8 // Number of sequences that are calculated concurrently
#define UPPER_BOUND SHRT_MAX

/**
 * v1.0: 
 *  - working with signed 16 bit values (short)
 *  - calculating column by column (not 4 of them at once)
 *  - db sequences are not padded
 *  - using saturated arithmetic
 * 
 * TODO: - maybe add support for ending sequence earlier if it can not be better than best result
 *       - align data! (But is it important for integers and shorts, or only for floats?)
 *       - use unsigned values to get bigger range (can that be done?)
 *       - use as less registers as possible! Investigate assembly code to determine how to do it.
 *       - check for overflow and use bigger precision in that case
 *       - maybe pad columns so length%4 == 0, that way I can check for sequence ending only on every 4th column, not every column
 */
class Swimd {
public:
    /**
     * Compares query sequence with each database sequence and returns similarity scores.
     * Sequences are not represented as arrays of letters, but as arrays of indices of letters in alphabet.
     * For example, if alphabet is {A,C,T,G} and sequence is ACCTCAG it will be represented as 0112103.
     * Opening of gap is penalized with gapOpen + gapExt, while gap extension is penalized with gapExt.
     * @param query Query sequence.
     * @param queryLength Length of query sequence.
     * @param db Array of database sequences (each sequence is also an array).
     * @param dbLength Number of database sequences.
     * @param dbSeqLengths Array of lengths of database sequences.
     * @param gapOpen
     * @param gapExt
     * @param scoreMatrix Matrix of dimensions (alphabetLength, alphabetLength).
     * @param alphabetLength
     * @return Largest similarity score for every database sequence. Null if overflow happens.
     */
    static vector<short> searchDatabase(Byte query[], int queryLength, Byte ** db, int dbLength, int dbSeqLengths[],
					int gapOpen, int gapExt, short ** scoreMatrix, int alphabetLength);

private:
    static bool loadNextSequence(int &nextDbSeqIdx, int dbLength, int &currDbSeqIdx, Byte * &currDbSeqPos, 
				 int &currDbSeqLength, Byte ** db, int dbSeqLengths[]);
};

#endif /* SWIMD_HPP */
