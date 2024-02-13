# Demo script for how to spawn an MTurk experiment.

python3 spawn_experiment.py \
  --experiment-name=202303 \
  --task-namespace=clip-resnet50_natural \
  --task-type=2afc \
  --n-tasks=63 \
  --n-repetitions=1 \
  --environment=real \
  --reward=2.79 \
  --row-variability-threshold=5 \
  --min-instruction-time=15 \
  --max-instruction-time=-1 \
  --min-total-response-time=135 \
  --max-total-response-time=2500 \
  --catch-trial-ratio-threshold=0.8 \
  --max-total-assignments=3 \
  --max-demo-trials-attempts=3 \
  --hit-lifetime=4.0 \
  --assignment-lifetime=2.0 \
  --previous-single-participation-qualifications 202303 \
  --single-participation-qualifications 202303-clip-resnet50_natural \
     202303 \
  --output-folder=output/202303/clip-resnet50_natural
