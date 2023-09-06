#! /bin/bash

### Need
#   gdc-client : https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
###


# UNZIP tcga data download
tar -zxvf zips.tar.gz

list_cancer=`ls zips`

for types in $list_cancer; do
    echo zips/${types}
    b="${types:0:-4}"

    # unzip GDC Manifest File
    unzip zips/${types} -d $b
    cd $b
    tar -zxf biospecimen.c*

    # expression profile download with GDC Manifest
    mkdir exp
    cd exp
    gdc-client download -m ../gdc_manifest_*
    gunzip */*.gz
    cd ..
    cd ..
done

