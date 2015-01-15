## Comparison with other aligners

Here are described results of speed comparison of OpAl with other aligners: SSW, SSEARCH(FASTA) and SWIPE.  
SSW, SSEARCH and SWIPE do only Smith-Waterman alignment so we compared only for SW.

Aligners were tested by quering sequences against UniProtKB/Swiss-Prot database (contains 541762 sequences).  
Database can be obtained from www.uniprot.org/downloads -> UniProtKB/Swiss-Prot.  
Specific sequence can also be obtained from www.uniprot.org by searching it by name (Search tab).

All aligners were tested with following parameters:
* number of threads = 1
* gap opening = 3
* gap extension = 1
* score matrix = BLOSUM50

Only scores were calculated (not alignments). Time spent to read sequences and database was not measured.

How aligners were called:
* SSW: `./ssw_test -p uniprot_sprot.fasta <query_file>`
* OpAl: `./opal_aligner -s <query_file> uniprot_sprot.fasta`
* SSEARCH: `./ssearch36 -d 0 -T 1 -p -f -3 -g -1 -s BL50 <query_file> uniprot_sprot.fasta`
* SWIPE: `./swipe -a 1 -p 1 -G 3 -E 1 -M BLOSUM50 -b 0 -i <query_file> -d uniprot_sprot` NOTE: database had to be preprocessed for SWIPE using _makeblastdb_

Following table shows how much time took for different sequences to be aligned against UniProtKB/Swiss-Prot database.
All times are in seconds. Test were performed on i7-4770K CPU @ 3.50GHz with 32GB RAM (AVX2 support).

|                     | O74807 | P19930 | Q3ZAI3 | P18080 |
|---------------------|--------|--------|--------|--------|
| **query length**    |   110  |   195  |   390  |   513  |
| **SSW**             |   9.0  |  16.6  |  25.8  |  31.0  |
| **_OpAl(SSE4.1)_**  |   8.7  |  12.2  |  20.2  |  25.5  |
| **_OpAl(AVX2)_**    |   5.2  |   6.9  |  10.8  |  14.7  |
| **SSEARCH**         |   7.0  |  11.7  |  18.3  |  22.4  |
| **SWIPE**           |   5.3  |   9.5  |  17.8  |  23.1  |
