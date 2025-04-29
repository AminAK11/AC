import numpy as np
import matplotlib.pyplot as plt
from brain import *

def binary_training():
    brain = Brain(p = 0.1, cap_size = 100, beta = 0.1, no_classes = 2, n = 1000, in_n = 500)
    choices = brain.binary_training(time_steps=80)    
    
    plt.figure(figsize=(10, 5))
    plt.plot(choices, marker='o')
    plt.title("Binary Training Choices")
    plt.xlabel("Time Steps")
    plt.ylabel("Choices")    
    plt.savefig("plots/binary_training.png")

def binary_training_test():
    correct = 0
    n = 10
    for i in range(n): 
        brain = Brain(p = 0.1, cap_size = 100, beta = 0.1, no_classes = 2, n = 1000, in_n = 500)
        p_A = np.random.rand()
        p_B = np.random.rand()
        choices = brain.binary_training(time_steps=800, p_A=p_A, p_B=p_B)
        
        print("testing with p_A: ", p_A, "p_B: ", p_B, "round: ", i)
        if p_B > p_A:
            c = np.sum(choices[-10:]) >= 9
            if not c: print("Wrong prediction ^^")
            correct += int(c)
        else:
            c = np.sum(choices[-10:]) == 0
            if not c: print("Wrong prediction ^^")
            correct += int(c)
        

    print("accuracy", correct / n)





if __name__ == '__main__':
    binary_training()
    # binary_training_test()