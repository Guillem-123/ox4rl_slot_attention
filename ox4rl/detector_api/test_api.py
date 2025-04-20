
from ox4rl import SPACEDetector
from ocatari.core import OCAtari
import random
import matplotlib.pyplot as plt

def test_SPACEDetector_using_OCAtari():
    game_name = "skiing"
    space_detector = SPACEDetector(game_name)
    env = OCAtari("ALE/Skiing-v5", mode="ram", hud=False, obs_mode="ori")
    observation, info = env.reset()
    action = random.randint(0, env.nb_actions-1)
    obs, reward, terminated, truncated, info = env.step(action)
    i = 0
    while not terminated:
        i += 1
        action = random.randint(0, env.nb_actions-1)
        obs, reward, terminated, truncated, info = env.step(action)
        objects = space_detector.detect(obs)
        print(objects)
        if i% 50 == 0:
            plt.imshow(obs)
            plt.show()


test_SPACEDetector_using_OCAtari()