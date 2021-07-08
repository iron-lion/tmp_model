#!/bin/bash
# Original works: https://github.com/hemberg-lab/scRNA.seq.datasets/

mkdir -p ./RAW;
cd ./RAW;

# get data
wget https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-5061/E-MTAB-5061.processed.1.zip
unzip E-MTAB-5061.processed.1.zip
head -n 1 pancreas_refseq_rpkms_counts_3514sc.txt > labels.txt
sed -i '1s/#samples//' labels.txt
sed -i '1s/#samples/#samples\tHNM/' pancreas_refseq_rpkms_counts_3514sc.txt
cut -f -3516 pancreas_refseq_rpkms_counts_3514sc.txt >  cut_exp.csv
# get metadata
wget https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-5061/E-MTAB-5061.sdrf.txt
sed -i 's/ /_/g' E-MTAB-5061.sdrf.txt
