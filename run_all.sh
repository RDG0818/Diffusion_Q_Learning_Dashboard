#!/bin/bash

ENVIRONMENTS=("ant" "hopper" "swimmer" "walker2d" "halfcheetah")
SEEDS=(1 2 3)

for ENV in "${ENVIRONMENTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    EXP_NAME="${ENV}-medium-expert-${SEED}"
    TEACHER_DIR="results/${EXP_NAME}"
    STUDENT_PATH="${TEACHER_DIR}/student_policy.pth"

    echo "============================"
    echo "Running main.py for ${EXP_NAME}"
    echo "============================"
    python main.py env=${ENV} seed=${SEED} exp=${EXP_NAME} quality=medium-expert

    echo "----------------------------"
    echo "Running distill.py for ${EXP_NAME}"
    echo "----------------------------"
    python distill.py \
      teacher_model_dir=${TEACHER_DIR} \
      student_model_save_path=${STUDENT_PATH} \
      env=${ENV} \
      quality=medium-expert \
      distill_epochs=200 \
      distill_batch_size=1024 \
      distill_lr=3e-4 \
      eval_episodes=10 \
      seed=41 \
      dagger_iters=25 \
      dagger_rollouts=50 \
      dagger_epochs=200 \
      weight_decay=1e-4

  done
done
