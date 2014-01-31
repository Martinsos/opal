## Comparison with other aligners

Here are described results of speed comparison of SWIMD with other aligners: SSW, SSEARCH(FASTA) and SWIPE.  
SSW, SSEARCH and SWIPE do only Smith-Waterman alignment so we compared only for SW.

Aligners were tested by quering sequences against UniProtKB/Swiss-Prot database (contains 541762 sequences).  
Sequences and database can be obtained from www.uniprot.org (www.uniprot.org/downloads -> UniProtKB/Swiss-Prot).

All aligners were tested with following parameters:
* only score is calculated (not alignments)
* time of reading sequences and database was not measured
* number of threads = 1
* gap opening = 3
* gap extension = 1
* score matrix = BLOSUM50

How aligners were called:
* SSW: `./ssw_test -p uniprot_sprot.fasta <query_file>`
* SWIMD: `./swimd_aligner -s <query_file> uniprot_sprot.fasta`
* SSEARCH: `./ssearch36 -d 0 -T 1 -p -f -3 -g -1 -s BL50 <query_file> uniprot_sprot.fasta`
* SWIPE: `./swipe -a 1 -p 1 -G 3 -E 1 -M BLOSUM50 -b 0 -i <query_file> -d uniprot_sprot` NOTE: database had to be preprocessed for SWIPE using _makeblastdb_

Following table shows how much time took for different sequences to be aligned against UniProtKB/Swiss-Prot database.
All time is in seconds. Tests were performed on Intel Core i3 M 350 @ 2.27GHz with 4GB RAM.

|                  | O74807 | P19930 | Q3ZAI3 | P18080 |
|------------------|--------|--------|--------|--------|
| **query length** |   110  |   195  |   390  |   513  |
| **SSW**          |  20.2  |  36.4  |  54.4  |  68.4  |
| **_SWIMD_**      |  18.2  |  26.6  |  46.0  |  60.5  |
| **SSEARCH**      |  16.8  |  27.3  |  38.6  |  48.2  |
| **SWIPE**        |   9.6  |  17.0  |  32.1  |  41.5  |
