#include "Swimd.h"

using namespace std;

SWIMD_SEARCH_DATABASE() {
    int resultCode;
    resultCode = swimdSearchDatabase8(query, queryLength,
				      db, dbLength, dbSeqLengths,
				      gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
    if (resultCode != 0) {
	resultCode = swimdSearchDatabase16(query, queryLength, 
					   db, dbLength, dbSeqLengths,
					   gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
	if (resultCode != 0) {
	    resultCode = swimdSearchDatabase32(query, queryLength, 
					       db, dbLength, dbSeqLengths,
					       gapOpen, gapExt, scoreMatrix, alphabetLength, scores);
	}
    }
    return resultCode;
}
