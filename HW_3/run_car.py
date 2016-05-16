# from HW_3.cars import *
from cars.world import SimpleCarWorld
from cars.agent import SimpleCarAgent
from cars.physics import SimplePhysics
from cars.track import generate_map
import numpy as np
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", type=int)
parser.add_argument("-f", "--filename", type=str)
parser.add_argument("-e", "--evaluate", action='store_true')
parser.add_argument("-k", "--kb-control", action='store_true')
parser.add_argument("-n", "--number", type=int)
parser.add_argument("--seed", type=int)
args = parser.parse_args()

print(args.steps, args.seed, args.filename, args.evaluate)

steps = args.steps
seed = args.seed if args.seed else 23
np.random.seed(seed)
random.seed(seed)
m = generate_map(8, 5, 3, 3)

w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
agent = SimpleCarAgent.from_file(args.filename) if args.filename else SimpleCarAgent()
agent.allow_kb_control = args.kb_control

agents = [agent] + [SimpleCarAgent() for i in range(1, args.number or 1)]


if args.evaluate:
    print(w.evaluate_agent(agents[0], args.steps or 1000))
else:
    w.set_agents(agents)
    w.run(args.steps)
