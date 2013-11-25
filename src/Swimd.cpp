#include "Swimd.hpp"

using namespace std;

SEARCH_DATABASE(Swimd::,) {
    try {
	return Swimd::searchDatabase8(query, queryLength, db, dbLength, dbSeqLengths,
				      gapOpen, gapExt, scoreMatrix, alphabetLength);
    } catch(DatabaseSearchOverflowException e) {
	try {
	    return Swimd::searchDatabase16(query, queryLength, 
					   db, dbLength, dbSeqLengths,
					   gapOpen, gapExt, scoreMatrix, alphabetLength);
	} catch(DatabaseSearchOverflowException e) {
	    // Not catching this one because there is nothing I can do
	    return Swimd::searchDatabase32(query, queryLength, 
					   db, dbLength, dbSeqLengths,
					   gapOpen, gapExt, scoreMatrix, alphabetLength);
	}
    }
}
