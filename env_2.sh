#!/bin/bash
module  purge 
source  /home/bingxing2/apps/package/pytorch/2.4.0+cu121_cp310/env.sh
module unload compilers/gcc/11.3.0
module load compilers/gcc/12.2.0
source  activate  L1
