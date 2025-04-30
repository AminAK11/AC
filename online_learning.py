import numpy as np
import matplotlib.pyplot as plt
from brain import *

def binary_training():
    brain = Brain(p = 0.1, cap_size = 100, beta = 0.1, no_classes = 3, n = 1000, in_n = 500)
    props = [0.33, 0.33, 0.33]
    choices = brain.binary_training(props, time_steps=100)

    fig, ax = plt.subplots(figsize=(20, 6))
    
    
    classes = ["A", "B", "C"]
    ax.set_yticks(range(len(classes)), labels=classes)
    
    fig.legend([str(props[0]), str(props[1]), str(props[2])])
    ax.imshow(choices, cmap="binary")
    ax.set_title(f"Game history \n P(A = 1) = {str(props[0])}  P(B = 1) = {str(props[1])}  P(C = 1) = {str(props[2])}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Choices")
    

    fig.savefig("plots/online_training.png")

def binary_training_test():
    correct = 0
    n = 10
    for _ in range(n): 
        brain = Brain(p = 0.1, cap_size = 200, beta = 0.1, no_classes = 3, n = 2000, in_n = 500)

        props = [np.random.rand() for _ in range(brain.no_classes)]
        choices = brain.binary_training(props, time_steps=400)
        print("props: ", props)
                
        best = np.argmax(props)
        c = np.sum(choices[best, -10:]) >= 9

        if c: 
            correct += 1
        else:
            print("Wrong prediction ^^")

    print("accuracy", correct / n)


if __name__ == '__main__':
    binary_training()
    binary_training_test()