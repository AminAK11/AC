import numpy as np
import matplotlib.pyplot as plt
from brain import *
from matplotlib.colors import ListedColormap, BoundaryNorm


def get_brain():
    return Brain(p = 0.1, cap_size = 100, beta = 0.1, no_classes = 2, n = 700, in_n = 1000)

def binary_training():
    brain = get_brain()
    props = [0.4, 0.6]
    choices = brain.binary_training(props, time_steps=100)

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    
    classes = [chr(ord("A") + i) for i in range(len(props))]
    ax.set_yticks(range(len(classes)), labels=classes)
    fig.legend([str(props[i]) for i in range(len(props))])
    ax.imshow(choices, cmap="binary")
    ax.set_title(f"Game history \n props: {props}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Choices")

    # # no delta
    # brain = get_brain()
    # choices_no_delta = brain.binary_training(props, time_steps=100)
    # fig.legend([str(props[i]) for i in range(len(props))])
    # ax[1].set_yticks(range(len(classes)), labels=classes)
    # ax[1].imshow(choices_no_delta, cmap="binary")
    # ax[1].set_title(f"Game history without delta \n props: {props}")
    # ax[1].set_xlabel("Time Steps")
    # ax[1].set_ylabel("Choices")
    
    fig.savefig("plots/online_training.png")

def binary_training_test():
    correct = 0
    n = 15
    for i in range(n): 
        brain = get_brain()

        props = [np.random.rand() for _ in range(brain.no_classes)]
        choices = brain.binary_training(props, time_steps=500)
        print("props: ", props)

        best = np.argmax(props)
        c = np.sum(choices[best, -50:]) >= 35

        if c: 
            correct += 1
        else:
            print("Wrong prediction ^^")
        
        
        plt.figure(figsize=(20, 10))
        
        classes = [chr(ord("A") + i) for i in range(len(props))]
        plt.yticks(range(len(classes)), labels=classes)
        plt.legend([str(props[i]) for i in range(len(props))])
        plt.imshow(choices, cmap="binary", aspect="3")
        plt.title(f"Game history \n props: {props}")
        plt.xlabel("Time Steps")
        plt.ylabel("Choices")
        
        plt.savefig(f"plots/sletigen/online_training_test_{i}.png")
        
    print("accuracy", correct / n)

def two_brains_binary_game(rounds=700):
    brain1 = Brain(p = 0.1, cap_size = 100, beta = 0.1, no_classes = 2, n = 2000, in_n = 1000)
    brain2 = Brain(p = 0.1, cap_size = 100, beta = 0.1, no_classes = 2, n = 2000, in_n = 1000)
    b1choices = np.zeros((brain1.no_classes, rounds))
    b2choices = np.zeros((brain2.no_classes, rounds))

    for i in range(rounds):
        in_class_activations1 = brain1.random_activations()
        in_class_activations2 = brain2.random_activations()
        
        activations_t_1_1 = brain1._get_activations(in_class_activations1) 
        activations_t_1_2 = brain2._get_activations(in_class_activations1)
        
        b1choice = brain1.get_choice(activations_t_1_1)
        b1choices[b1choice, i] = 1
        b2choice = brain2.get_choice(activations_t_1_2)
        b2choices[b2choice, i] = 1

        if b1choice == b2choice:
            #b1 wins
            brain1.update_weights(in_class_activations1, activations_t_1_1, b1choice)
            brain2.update_weights(in_class_activations2, activations_t_1_2, b1choice, negative=True)
            b1choices[b1choice, i] = 0.7
        else:
            #b2 wins
            brain2.update_weights(in_class_activations2, activations_t_1_2, b2choice)
            brain1.update_weights(in_class_activations1, activations_t_1_1, b1choice, negative=True)

            b2choices[b2choice, i] = 0.7
    
    cmap = ListedColormap(["white", "yellowgreen", "gold"])
    bounds = [0, 0.1, 0.71, 1.01]
    norm = BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(2, 1, figsize=(50, 6))
    classes = ["A", "B"]
    
    ax[0].imshow(b1choices, cmap=cmap, norm=norm)
    ax[0].set_title("Brain 1")
    ax[0].set_yticks(range(len(classes)), labels=classes)
    ax[0].set_xlabel("Time Steps")
    ax[0].set_ylabel("Choices")

    ax[1].imshow(b2choices, cmap=cmap, norm=norm)
    ax[1].set_title("Brain 2")
    ax[1].set_yticks(range(len(classes)), labels=classes)
    ax[1].set_xlabel("Time Steps")
    ax[1].set_ylabel("Choices")

    fig.savefig("plots/braingames.png")


if __name__ == '__main__':
    binary_training()
    # binary_training_test()
    # two_brains_binary_game()