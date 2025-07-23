#!/bin/bash

#job_id
job_id=$1
echo "Job ID: "${job_id}
echo $job_id > job_id.txt

python3 05_1_Fit_Dpi_varying_gamma.py --date 2025_07_02
python3 05_2_Full_Toys_B2Dpi_misID_varying_gamma.py --date 2025_07_02

