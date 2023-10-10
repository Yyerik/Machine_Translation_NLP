There are three python programs here (`-h` for usage):

- `./align_ibm1` ibm1 word alignment words.

- `./align_ibm2` ibm2 word alignment words.

- `./check-alignments` checks that the entire dataset is aligned, and
  that there are no out-of-bounds alignment points.

- `./score-alignments` computes alignment error rate.

The commands work in a pipeline. For instance:

   > ./align -t 0.9 -n 1000 | ./check | ./grade -n 5

The `data` directory contains a fragment of the Canadian Hansards,
aligned by Ulrich Germann:

- `hansards.e` is the English side.

- `hansards.f` is the French side.

- `hansards.a` is the alignment of the first 37 sentences. The 
  notation i-j means the word as position i of the French is 
  aligned to the word at position j of the English. Notation 
  i?j means they are probably aligned. Positions are 0-indexed.

  # Machine-Translation-HW2

The IBM model 1 Expectation Maximization algorithm is implemented in align_ibm_model1 file.

To run the IBM model 1, simply run 

    python align_ibm_model1 -n [number of sentences] > [alignment output]

An example usage is 

    python align_ibm_model1 -n 10000 > alignment_ibm_model1

to run the IBM Model 1 on 10000 sentences and save the alignment output to `alignment_ibm_model1` file.

To score the alignments and compute the AER, simply run as usual

    python score-alignments < [alignment_output_file]

For this example, simply run 
    
    python score-alignments < alignment_ibm_model1

## Extension: IBM Model 2
The IBM model 1 Expectation Maximization algorithm is implemented in align_ibm2 file.

To run the IBM model 2, simply run 

    python align_ibm2 -n [number of sentences] > [alignment output]

An example usage is 

    python align_ibm2 -n 10000 > alignment

to run the IBM Model 2 on 10000 sentences and save the alignment output to `alignment` file.

To score the alignments and compute the AER, simply run as usual

    python score-alignments < [alignment_output_file]

For this example, simply run 
    
    python score-alignments < alignment

Note the `alignment` file contains the alignments produced by the IBM Model 2 when trained on `n = 100,000` sentences. We achieved an AER of about 0.28. 

 

