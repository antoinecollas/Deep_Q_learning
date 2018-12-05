# Deep_Q_learning
Implementation of "Human-level control through deep reinforcement learning" (Mnih et al.): https://www.nature.com/articles/nature14236

# References:
Mnih et al.: https://www.nature.com/articles/nature14236
Seita: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

# TODO:
- states should overlap (https://github.com/openai/gym/issues/275) (OK)
- verify loss backward must not be on Q_hat ! (OK)
- replace mean reward by sum of rewards (OK)
- increase batch size (OK)
- print less and store less in tensorboard (10x less) (OK)
- sometimes test_get_training_data.py failes (OK)
- render and visualize some episodes (OK)
- NATURE paper uses frames as unit in hyperparameters whereas this implementation uses episodes (or a mixture of both ??)
- improve modularity:
    - create train function (OK)
    - create file that uses this function to train breakout (OK)
    - put an example with cartpole (which seems to be easier to train than breakout) (OK)
- organize repo (OK)
- parallelize environment: do 4 environments in parallel in order to update every step and not every 4 steps.
- add unit test on load/save with cartpole
- colors are inverted in tensorboard
- compare speed with openAI baselines
- add gpu utilization in tensorboard (https://github.com/lanpa/tensorboardX/blob/master/examples/demo_nvidia_smi.py)
