# Lyft Motion Prediction for Autonomous Vehicles

[Kaggle compete](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles)


## TODO LIST

- [ ] Run baseline for cnn model
- [ ] Add traffic light
- [ ] Predict confidence for regression model
- [ ] Run baseline for cnn-rnn model
- [ ] Multi-model


## Training

* Support auto resume


## Test


## Scripts

```shell
# Run interactive
srun -q=q_msai --mem=8G --gres=gpu:1 --pty bash -i

# Submit a job
sbatch slurm_run.sh
```
