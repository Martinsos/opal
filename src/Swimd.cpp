#include "Swimd.hpp"

using namespace std;

SEARCH_DATABASE(Swimd::,) {
    int resultCode;
    resultCode = Swimd::searchDatabase8(query, queryLength,
					db, dbLength, dbSeqLengths,
					gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
    if (resultCode != 0) {
	resultCode = Swimd::searchDatabase16(query, queryLength, 
					     db, dbLength, dbSeqLengths,
					     gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
	if (resultCode != 0) {
	    resultCode = Swimd::searchDatabase32(query, queryLength, 
						 db, dbLength, dbSeqLengths,
						 gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
	}
    }
    return resultCode;
}
