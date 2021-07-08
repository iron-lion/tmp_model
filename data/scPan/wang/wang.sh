#!/bin/bash
# Original works: https://github.com/hemberg-lab/scRNA.seq.datasets/

mkdir -p ./RAW;
cd RAW;
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE83nnn/GSE83139/suppl/GSE83139_tbx-v-f-norm-ntv-cpms.csv.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE83nnn/GSE83139/matrix/GSE83139-GPL11154_series_matrix.txt.gz
wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE83nnn/GSE83139/matrix/GSE83139-GPL16791_series_matrix.txt.gz

gunzip GSE83139_tbx-v-f-norm-ntv-cpms.csv.gz
gunzip GSE83139-GPL11154_series_matrix.txt.gz
gunzip GSE83139-GPL16791_series_matrix.txt.gz

head -n 38 GSE83139-GPL11154_series_matrix.txt | tail -1 > label.txt
head -n 50 GSE83139-GPL11154_series_matrix.txt | tail -1 >> label.txt

head -n 38 GSE83139-GPL16791_series_matrix.txt | tail -1 > label2.txt
head -n 50 GSE83139-GPL16791_series_matrix.txt | tail -1 >> label2.txt

sed -i 's/curated-cell-type: //g' label*
sed -i 's/!//g' label*
sed -i 's/"//g' label*
sed -i 's/ /_/g' label*
