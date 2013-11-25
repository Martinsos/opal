#ifndef SWIMD_HPP
#define SWIMD_HPP

#include <vector>
#include <exception>
#include <stdexcept>

using namespace std;

typedef unsigned char Byte; // TODO: name this uchar?


/**
 * v1.0: 
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
 *       - do some serious profiling
 *       - exchange #ifdefs with #if ?
 *       - add throw(smt) on right places
 *       - does SSE overflow wrap?
 *       - test overflow detection, especially for integers
 *       - think of some intelligent way on deciding which precision to use when
 *           - IMPORTANT! basic idea: return map instead of vector, use partial results when overflow happens
 *           - another idea: dont stop when overflow but go to next sequence, and report all which had overflow
 *           - third idea: do calculation like in basic idea but in blocks, so smaller precisions get more chance
 */
class Swimd {
public:
    /**
     * Compares query sequence with each database sequence and returns similarity scores.
     * Sequences are not represented as arrays of letters, but as arrays of indices of letters in alphabet.
     * For example, if alphabet is {A,C,T,G} and sequence is ACCTCAG it will be represented as 0112103.
     * Opening of gap is penalized with gapOpen, while gap extension is penalized with gapExt.
     * gapOpen, gapExt and scores from scoreMatrix must be in (INT_MIN/2, INT_MAX/2).
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
#define SEARCH_DATABASE(PREFIX,SUFFIX)					\
    vector<int> PREFIX searchDatabase ## SUFFIX				\
    (Byte query[], int queryLength, Byte ** db, int dbLength, int dbSeqLengths[], \
     int gapOpen, int gapExt, int ** scoreMatrix, int alphabetLength)
    
    /**
     * Main entry function for searching database.
     * It calls implementations for specific score precisions and tries to use lowest precision possible.
     */
    static SEARCH_DATABASE(,);    // searchDatabase(..)

    /**
     * Implementations that use different score precision.
     * Number of concurrently calculated sequences is 128 / precision,
     * therefore less precision is more speed.
     * If score overflow happens, exception is thrown.
     */
    static SEARCH_DATABASE(, 8);  // searchDatabase8(..)
    static SEARCH_DATABASE(, 16); // searchDatabase16(..)
    static SEARCH_DATABASE(, 32); // searchDatabase32(..)

    class DatabaseSearchOverflowException;

private:
    static bool loadNextSequence(int &nextDbSeqIdx, int dbLength, int &currDbSeqIdx, Byte * &currDbSeqPos, 
				 int &currDbSeqLength, Byte ** db, int dbSeqLengths[]);
};


inline bool Swimd::loadNextSequence(int &nextDbSeqIdx, int dbLength, int &currDbSeqIdx, Byte * &currDbSeqPos, 
				    int &currDbSeqLength, Byte ** db, int dbSeqLengths[]) {
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

/**
 * This exception is thrown when score overflow happens in database search.
 */
class Swimd::DatabaseSearchOverflowException : public exception {
public:
    DatabaseSearchOverflowException(const char* msg = "Overflow detected in database search") {
	this->msg = msg;
    }
    ~DatabaseSearchOverflowException() throw() {};

    const char* what() {
	return this->msg;
    }	

private:
    const char* msg;
};


#endif /* SWIMD_HPP */
