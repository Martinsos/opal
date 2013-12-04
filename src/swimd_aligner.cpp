#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <ctime>

#include "Swimd.h"
#include "ScoreMatrix.h"

using namespace std;

int readFastaSequences(const char* path, char* alphabet, int alphabetLength, vector< vector<unsigned char> >* seqs);

int main(int argc, char * const argv[]) {
    int gapOpen = 3;
    int gapExt = 1;
    ScoreMatrix scoreMatrix = getBlosum50();
    
    //----------------------------- PARSE COMMAND LINE ------------------------//
    char* scoreMatrixFilepath = 0;
    bool silent = false;
    int option;
    while ((option = getopt(argc, argv, "o:e:m:s")) >= 0) {
        switch (option) {
        case 'o': gapOpen = atoi(optarg); break;
        case 'e': gapExt = atoi(optarg); break;
        case 'm': strcpy(scoreMatrixFilepath, optarg); break;
        case 's': silent = true; break;
        }
    }
    if (optind + 2 != argc) {
        fprintf(stderr, "\n");
        fprintf(stderr, "Usage: swimd_aligner [options...] <query.fasta> <db.fasta>\n");        
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "\t-go N\tN is gap opening penalty. [default: 3]\n");
        fprintf(stderr, "\t-e N\tN is gap extension penalty. [default: 1]\n");
        fprintf(stderr, "\t-m FILE\tFILE contains score matrix and some additional data. [default: Blosum50]\n");
        fprintf(stderr, "\t-s\tIf specified, there will be no score output (silent mode).\n");
        return 1;
    }
    //-------------------------------------------------------------------------//

    if (scoreMatrixFilepath != 0) {
        // TODO: read matrix from file and store it to scoreMatrix
    }

    int readResult;
    // Build query
    char* queryFilepath = argv[optind];
    vector< vector<unsigned char> >* querySequences = new vector< vector<unsigned char> >();
    printf("Reading query fasta file...\n");
    readResult = readFastaSequences(queryFilepath, scoreMatrix.alphabet, scoreMatrix.alphabetLength, querySequences);
    if (readResult) {
        printf("Error: There is no file with name %s\n", queryFilepath);
        return 1;
    }
    unsigned char* query = (*querySequences)[0].data();
    int queryLength = (*querySequences)[0].size();

    // Build db
    char* dbFilepath = argv[optind+1];    
    vector< vector<unsigned char> >* dbSequences = new vector< vector<unsigned char> >();
    printf("Reading database fasta file...\n");
    readResult = readFastaSequences(dbFilepath, scoreMatrix.alphabet, scoreMatrix.alphabetLength, dbSequences);
    if (readResult) {
        printf("Error: There is no file with name %s\n", dbFilepath);
        return 1;
    }
    int dbLength = dbSequences->size();
    unsigned char* db[dbLength];
    int dbSeqLengths[dbLength];
    for (int i = 0; i < dbSequences->size(); i++) {
        db[i] = (*dbSequences)[i].data();
        dbSeqLengths[i] = (*dbSequences)[i].size();
    }

    printf("Searching...\n");
    // ----------------------------- MAIN CALCULATION ----------------------------- //
    int* scores = new int[dbLength];
    clock_t start = clock();
    int resultCode = swimdSearchDatabase(query, queryLength, db, dbLength, dbSeqLengths,
                                         gapOpen, gapExt, scoreMatrix.matrix, scoreMatrix.alphabetLength,
                                         scores);
    clock_t finish = clock();
    double cpuTime = ((double)(finish-start))/CLOCKS_PER_SEC;
    // ---------------------------------------------------------------------------- //
    
    if (!silent) {
        printf("Scores: \n");
        for (int i = 0; i < dbLength; i++)
            printf("%d ", scores[i]);
        printf("\n");
    }

    printf("Cpu time of searching: %lf\n", cpuTime);

    // Free allocated space
    delete querySequences;
    delete dbSequences;
    delete[] scores;
    
    return 0;
}


/** Reads sequences from fasta file.
 * @param [in] path Path to fasta file containing sequences.
 * @param [in] alphabet
 * @param [in] alphabetLength
 * @param [out] seqs Sequences will be stored here, each sequence as vector of indexes from alphabet.
 * @return 0 if all ok, positive number otherwise.
 */
int readFastaSequences(const char* path, char* alphabet, int alphabetLength, vector< vector<unsigned char> >* seqs) {
    seqs->clear();
    
    FILE* file = fopen(path, "r");
    if (file == 0)
        return 1;

    unsigned char letterIdx[128]; //!< letterIdx[c] is index of letter c in alphabet
    for (int i = 0; i < alphabetLength; i++)
        if (alphabet[i] == '*') { // '*' represents all characters not in alphabet
            for (int j = 0; j < 128; j++)
                letterIdx[j] = i;
            break;
        }
    for (int i = 0; i < alphabetLength; i++)
        letterIdx[alphabet[i]] = i;

    bool inHeader = false;
    bool inSequence = false;
    int buffSize = 4096;
    char buffer[buffSize];
    while (!feof(file)) {
        int read = fread(buffer, sizeof(char), buffSize, file);
        for (int i = 0; i < read; ++i) {
            char c = buffer[i];
            if (inHeader) { // I do nothing if in header
                if (c == '\n')
                    inHeader = false;
            } else {
                if (c == '>') {
                    inHeader = true;
                    inSequence = false;
                } else {
                    if (c == '\r' || c == '\n')
                        continue;
                    // If starting new sequence, initialize it.
                    if (inSequence == false) {
                        inSequence = true;
                        seqs->push_back(vector<unsigned char>());
                    }
                    //
                    seqs->back().push_back(letterIdx[c]);
                }
            }
        }
    }

    fclose(file);
}
 

