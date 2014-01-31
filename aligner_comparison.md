## Comparison with other aligners

Here are described results of speed comparison of SWIMD with other aligners: SSW, SSEARCH(FASTA) and SWIPE.
SSW, SSEARCH and SWIPE do only Smith-Waterman alignment so we compared only for SW.

Aligners were tested by quering sequences against UniProtKB/Swiss-Prot database.
All of them can be obtained from www.uniprot.org.

All aligners were tested with following parameters:
* only score is calculated (not alignments)
* time of reading sequences and database was not measured
* number of threads = 1
* gap opening = 3
* gap extension = 1
* score matrix = BLOSUM50

Following table shows how much time took for different sequences to be aligned againts UniProtKB/Swiss-Prot database.
All time is in seconds.

| Aligner | O74807 | Q3ZAI3 | P18080 |
|---------|--------|--------|--------|
| SSW     |  20.4  |  55.2  |     ?  |
| SWIPE   |   9.6  |  32.3  |  41.9  |
| SSEARCH |  16.0  |  38.8  |  48.6  |
| SWIMD   |     ?  |     ?  |     ?  |
