import gym, argparse
from deepq.wrapper_gym import SkipFrames
from deepq.deepq import train_deepq
from deepq.neural_nets import CNN2
from deepq.utils import preprocess

def main(env_name):
    AGENT_HISTORY_LENGTH = 4
    env = gym.make(env_name)
    env = SkipFrames(env, AGENT_HISTORY_LENGTH-1, preprocess)
    Q_network = CNN2(AGENT_HISTORY_LENGTH, env.action_space.n)

    train_deepq(
        env=env,
        env_name=env_name,
        Q_network=Q_network,
        input_as_images=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a machine translation.')
    parser.add_argument("-e", dest="env", required=True,
    help="Atari environment: becareful it must a NoFrameSkip environment!! (Breakout, Pong, ...)")
    args = parser.parse_args()
    main(str(args.env)+'NoFrameskip-v4')