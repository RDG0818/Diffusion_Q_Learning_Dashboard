#!/bin/bash

ENVS=("halfcheetah" "walker2d" "hopper")
DATASETS=("medium" "medium-expert" "expert")
SEEDS=(1 2)

TOTAL_EXPERIMENTS=$((${#ENVS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))
CURRENT_EXPERIMENT=0

for ENV in "${ENVS[@]}"
do
  for DATASET in "${DATASETS[@]}"
  do
    for SEED in "${SEEDS[@]}"
    do

      ((CURRENT_EXPERIMENT++))

      ENV_NAME="${ENV}-${DATASET}-v2"
      
      echo "----------------------------------------------------------------"
      echo "--- Starting Experiment ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}: ${ENV_NAME} (Seed: ${SEED})"
      echo "----------------------------------------------------------------"
      
      python main.py --env_name "$ENV_NAME" --seed "$SEED"
      
      echo "--- Finished Experiment: ${ENV_NAME} (Seed: ${SEED}) ---"
      echo "" 
    done
  done
done

echo "================================================================="
echo "All experiments completed."
echo "================================================================="

