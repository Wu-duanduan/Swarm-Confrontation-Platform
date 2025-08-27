# Swarm Confrontation Platform
A Simulation Platform for MARL Training and Evaluation in Swarm Confrontation
In swarm confrontation, robots must swiftly formulate strategies in transient environments, a challenge well-suited for multi-agent reinforcement learning (MARL). However, existing confrontation platforms suffer from the lack of sufficient diversity in models and scalable frameworks, hindering MARL's widespread applications. We introduce a novel platform for training, simulating, and evaluating MARL algorithms in complex swarm confrontation scenarios. It integrates robot, environment, and rule models to simulate complex confrontation scenarios. Equipped with a decentralized task allocator and path planner for each robot, the platform enables scalable cooperation across dynamic environments. Extensive experiments demonstrate that our platform effectively simulates confrontations involving up to twenty agents per side, providing robust algorithm selection strategies for diverse scenarios.

# Installation instructions:
The code has the following requirements:
1. Simulation_python: Python 3.8 
2. Simulation_java: Java and Python 3.8

We include a requirements.txt file as a reference, but note that such a file includes more libraries than the ones strictly needed to run our code.

# How to run the code:
To run the code, move to the Simulation_python/Train folder and executes the following files:
1. IQL: python train_IQL.py
2. PS_IQL: python train_PS_IQL.py
3. VDN: python train_VDN.py
4. COMA: python train_COMA.py
5. QMIX: python train_QMIX.py
   
After training, you can test the trained models by executing the following files:
1. IQL: python test_IQL.py
2. PS_IQL: python test_PS_IQL.py
3. VDN: python test_VDN.py
4. COMA: python test_COMA.py
5. QMIX: python test_QMIX.py
