#!/bin/bash
for IDX in {1000..1010}
do
    sbatch synthetic_run.sh $IDX
done
