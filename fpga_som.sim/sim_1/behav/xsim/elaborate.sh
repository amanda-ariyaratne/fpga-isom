#!/bin/bash -f
# ****************************************************************************
# Vivado (TM) v2020.2 (64-bit)
#
# Filename    : elaborate.sh
# Simulator   : Xilinx Vivado Simulator
# Description : Script for elaborating the compiled design
#
# Generated by Vivado on Sat Dec 04 15:00:46 +0530 2021
# SW Build 3064766 on Wed Nov 18 09:12:47 MST 2020
#
# Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
#
# usage: elaborate.sh
#
# ****************************************************************************
set -Eeuo pipefail
# elaborate design
echo "xelab -wto c97a37bdf6864dd9951b028e3f1247ee --incr --debug typical --relax --mt 8 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip --snapshot csom_tb_behav xil_defaultlib.csom_tb xil_defaultlib.glbl -log elaborate.log"
xelab -wto c97a37bdf6864dd9951b028e3f1247ee --incr --debug typical --relax --mt 8 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip --snapshot csom_tb_behav xil_defaultlib.csom_tb xil_defaultlib.glbl -log elaborate.log

