#!/bin/bash

# List of AugDirect values --MakeImbalance
#net_values="DiGSymib DiGSymCatib  DiG DiGib  DiGSymCatMixib"
# DiGi4 DiGu3 DiGu4
#addSym Sym addSympara
#Mag MagQin Sig Qua
#GCN GAT APPNP GIN Cheb SAGE
#JKNet pgnn mlp sgc"Cheb MagQin  DiGSymib DiGSymCatib  DiG DiGib  DiGSymCatMixib DiGSymCatMixSymib
net_values="DiGSymib"
q_value=1
Aug_value=0

layer_values=" 2 3 4 5"    #

Direct_dataset='telegram/telegram'  # Update your Direct_dataset value
Direct_dataset_filename=$(echo $Direct_dataset | sed 's/\//_/g')
unDirect_data='Cora'
generate_timestamp() {
  date +"%d%H%Ms%S"
}
timestamp=$(generate_timestamp)

# Iterate over each net value   --MakeImbalance
for layer in $layer_values; do
  logfile="outforlayer${layer}.log"  # Adjust log file name with layer number
    exec > $logfile 2>&1  # Redirect stdout and stderr to log file
  # Iterate over each layer value
  for net in $net_values; do
    nohup python3 All2MainStop.py --AugDirect=$Aug_value --net=$net   --IsDirectedData  --all1   --to_undirected\
    --layer=$layer  --q=$q_value  --Direct_dataset="$Direct_dataset"  --undirect_dataset="$unDirect_data" \
      > wrongname_${Direct_dataset_filename}Bala_Undirect_${timestamp}_Aug${Aug_value}${net}_layer${layer}q${q_value}.log &
    pid=$!

    wait $pid
  done
done