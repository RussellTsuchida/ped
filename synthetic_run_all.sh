#!/bin/bash
for IDX in {1..100}
do
    sbatch synthetic_run.sh $IDX
done
