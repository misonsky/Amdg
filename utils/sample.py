#coding=utf-8

import random
import os


seed = 12345678
random.seed(seed)

propotions=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]

path ="datasets/ecomm/train.txt"

examples = []

with open(path,"r",encoding="utf-8") as f:
    for line in f:
        line = line.strip().rstrip()
        examples.append(line)

for propo in propotions:
    num_sample = int(len(examples) * propo)
    samples = random.sample(examples,num_sample)
    new_dir = os.path.dirname(path)+"_%d"%(propo*100)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    with open(os.path.join(new_dir,"train.txt"),"w",encoding="utf-8") as f:
        for line in samples:
            line = line.strip().rstrip()
            f.write(line+"\n")



