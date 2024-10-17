#!/bin/sh
echo Hello from $USER on $HOSTNAME
macro=/software/rj23972/safety_net/tfpcbpggsz/local-python/run
script=/software/rj23972/safety_net/tfpcbpggsz/canorman_InvMassFit/invariant_mass_fit.py

GPU=5

$macro python3 $script --index 501 --gpu $GPU
