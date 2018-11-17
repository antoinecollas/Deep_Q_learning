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
- NATURE paper uses frames as unit in hyperparameters whereas this implementation uses episodes (or a mixture of both ??)
- compare speed with openAI baselines
- render and visualize some episodes
