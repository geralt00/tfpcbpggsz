#!/bin/bash

#job_id
job_id=$1
echo "Job ID: "${job_id}
echo $job_id > job_id.txt

python3 01_No_Efficiency.py --date 2025_06_24

