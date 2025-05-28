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



def two_brains_binary_game(rounds=500, version=1):
    if version == 1:
        brain_game_1(rounds)
    elif version == 2:
        brain_game_2(rounds)
    else:
        raise ValueError("Version must be 1 or 2")

def brain_game_1(rounds=500):
    '''
    Matching-pennies game - 0/1 reward
    '''
    
    brain1 = Brain(p = 0.1, cap_size = 200, beta = 0.1, no_classes = 2, n = 2000, in_n = 1000)
    brain2 = Brain(p = 0.1, cap_size = 200, beta = 0.1, no_classes = 2, n = 2000, in_n = 1000)
    b1choices = np.zeros((brain1.no_classes, rounds))
    b2choices = np.zeros((brain2.no_classes, rounds))
    
    freq_a_b1 = []
    freq_b_b1 = []
    
    prob_array_a = []
    prob_array_b = []
    for i in range(rounds):
        choices = []
        for _ in range(1):
            in_class_activations1 = brain1.random_activations()
            in_class_activations2 = brain2.random_activations()
            
            activations_t_1_1 = brain1._get_activations(in_class_activations1) 
            activations_t_1_2 = brain2._get_activations(in_class_activations2)
            choices.append(brain1.get_choice(activations_t_1_1))
            
        prob_array_b.append(sum(choices) / len(choices))
        prob_array_a.append(1 - prob_array_b[i])
        
        b1choice = brain1.get_choice(activations_t_1_1)
        b1choices[b1choice, i] = 1
        b2choice = brain2.get_choice(activations_t_1_2)
        b2choices[b2choice, i] = 1

        if b1choice == b2choice:
            #b1 wins
            brain1.update_weights(in_class_activations1, activations_t_1_1, b1choice)
            brain2.update_weights(in_class_activations2, activations_t_1_2, b1choice, negative=True)
            #b1choices[b1choice, i] = 0.7
        else:
            #b2 wins
            brain2.update_weights(in_class_activations2, activations_t_1_2, b2choice)
            brain1.update_weights(in_class_activations1, activations_t_1_1, b1choice, negative=True)

            #b2choices[b2choice, i] = 0.7
            
    
        freq_a_b1.append(np.sum(b1choices[0]) / (i + 1))
        freq_b_b1.append(1 - freq_a_b1[i])
    
    
    b1_a = np.sum(b1choices[0]) / len(b1choices[0])
    b1_b = np.sum(b1choices[1]) / len(b1choices[1])
    b2_a = np.sum(b2choices[0]) / len(b2choices[0])
    b2_b = np.sum(b2choices[1]) / len(b2choices[1])
    print(f'Brain 1: {b1_a} and {b1_b}')
    print(f'Brain 2: {b2_a} and {b2_b}')

    labels = ["p1:A      p2:A", "p1:B      p2:B"]
    x = np.arange(len(labels))

    width = 0.15

    plt.bar(x - 0.1, [b1_a, b1_b], width=width, color="lightblue", label="Brain 1", align="center", edgecolor="black")
    plt.bar(x + 0.1, [b2_a, b2_b], width=width, color="lightgreen", label="Brain 2", align="center", edgecolor="black")
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    

    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.xticks(x, labels)
    plt.ylim(0, 1)

    plt.legend()

    plt.savefig("plots/barplot.png")   
    
    cmap = ListedColormap(["white", "yellowgreen", "gold"])
    bounds = [0, 0.1, 0.71, 1.01]
    norm = BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(3, 1, figsize=(25, 6))
    classes = ["A", "B"]
    
    
    ax[0].imshow(b1choices, cmap=cmap, norm=norm)
    ax[0].set_title("Brain 1", fontsize=20)
    ax[0].set_yticks(range(len(classes)), labels=classes, fontsize=20)
    ax[0].set_xlabel("Time Steps", fontsize=25)
    ax[0].set_ylabel("Choices", fontsize=25)
    
    ax[1].axis("off")

    ax[2].imshow(b2choices, cmap=cmap, norm=norm)
    ax[2].set_title("Brain 2", fontsize=20)
    ax[2].set_yticks(range(len(classes)), labels=classes, fontsize=20)
    ax[2].set_xlabel("Time Steps", fontsize=25)
    ax[2].set_ylabel("Choices", fontsize=25)


    fig.savefig("plots/braingames.png")
    
    fig, ax = plt.subplots()
    ax.plot(freq_a_b1, color="black", label="Frequency of A in Brain 1")
    ax.plot(freq_b_b1, color="blue", label="Frequency of B in Brain 1")
    fig.savefig("plots/frequency_braingame.png")
    
    fig, ax = plt.subplots()
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.plot(prob_array_a, color="black")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Probability")
    # ax.plot(prob_array_b, color="blue", label="Probability of B in Brain 1")
    fig.savefig("plots/probability_braingame.png")

def brain_game_2(rounds):
    '''
    Matching-pennies game - alternative rewards
    '''
    
    payoff = np.array([[2, 1],
                        [1, 1]])
    
    brain1 = Brain(p = 0.1, cap_size = 200, beta = 0.1, no_classes = 2, n = 2000, in_n = 1000)
    brain2 = Brain(p = 0.1, cap_size = 200, beta = 0.1, no_classes = 2, n = 2000, in_n = 1000)
    b1choices = np.zeros((brain1.no_classes, rounds))
    b2choices = np.zeros((brain2.no_classes, rounds))

    for i in range(rounds):
        in_class_activations1 = brain1.random_activations()
        in_class_activations2 = brain2.random_activations()
        
        activations_t_1_1 = brain1._get_activations(in_class_activations1)
        activations_t_1_2 = brain2._get_activations(in_class_activations2)
        
        b1choice = brain1.get_choice(activations_t_1_1)
        b1choices[b1choice, i] = 1
        b2choice = brain2.get_choice(activations_t_1_2)
        b2choices[b2choice, i] = 1

        #if b1choice == b2choice:
            #b1 wins
            #b1choices[b1choice, i] = 0.7
        #else:
            #b2 wins
            #b2choices[b2choice, i] = 0.7
        
        #in_class_activations1 = brain1.random_activations(reward=v1)
        #in_class_activations2 = brain2.random_activations(reward=v2)
        
        update_activations1 = brain1.section_activations(in_class_activations1, b1choice, neuron_pr_class=brain1.n / brain1.no_classes)
        update_activations2 = brain2.section_activations(in_class_activations2, b2choice, neuron_pr_class=brain2.n / brain2.no_classes)
        
        for _ in range(payoff[b1choice, b2choice]):
            if (b1choice == b2choice):
                brain1.update_weights(in_class_activations1, update_activations1, b1choice)
                brain2.update_weights(in_class_activations2, update_activations2, b2choice, negative=True)
            else:
                brain1.update_weights(in_class_activations1, update_activations1, b1choice, negative=True)
                brain2.update_weights(in_class_activations2, update_activations2, b2choice)
                
        if i % 50 == 0:
            brain1.weights = brain1.weights / brain1.weights.sum(axis=0)
            brain2.weights = brain2.weights / brain2.weights.sum(axis=0)

    b1_a = np.sum(b1choices[0]) / len(b1choices[0])
    b1_b = np.sum(b1choices[1]) / len(b1choices[1])
    b2_a = np.sum(b2choices[0]) / len(b2choices[0])
    b2_b = np.sum(b2choices[1]) / len(b2choices[1])
    print(f'Brain 1: {b1_a} and {b1_b}')
    print(f'Brain 2: {b2_a} and {b2_b}')
    
    labels = ["p1:A      p2:A", "p1:B      p2:B"]
    x = np.arange(len(labels))

    width = 0.15

    plt.bar(x - 0.1, [b1_a, b1_b], width=width, color="lightblue", label="Brain 1", align="center", edgecolor="black")
    plt.bar(x + 0.1, [b2_a, b2_b], width=width, color="lightgreen", label="Brain 2", align="center", edgecolor="black")
    plt.axhline(y=0.4, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=0.6, color='black', linestyle='--', linewidth=1)
    

    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.xticks(x, labels)
    plt.ylim(0, 1)

    plt.legend()

    plt.savefig("plots/barplot2.png")   

    fig, ax = plt.subplots(3, 1, figsize=(25, 6))
    classes = ["A", "B"]    
            
    cmap = ListedColormap(["white", "yellowgreen", "gold"])
    bounds = [0, 0.1, 0.71, 1.01]
    norm = BoundaryNorm(bounds, cmap.N)
    
    ax[0].imshow(b1choices, cmap=cmap, norm=norm)
    ax[0].set_title("Brain 1", fontsize=20)
    ax[0].set_yticks(range(len(classes)), labels=classes, fontsize=20)
    ax[0].set_xlabel("Time Steps", fontsize=25)
    ax[0].set_ylabel("Choices", fontsize=25)
    
    ax[1].axis("off")

    ax[2].imshow(b2choices, cmap=cmap, norm=norm)
    ax[2].set_title("Brain 2", fontsize=20)
    ax[2].set_yticks(range(len(classes)), labels=classes, fontsize=20)
    ax[2].set_xlabel("Time Steps", fontsize=25)
    ax[2].set_ylabel("Choices", fontsize=25)

    fig.savefig(f"plots/braingames_2.png")


if __name__ == '__main__':
    plt.close()
    # binary_training()
    # binary_training_test()
    two_brains_binary_game(rounds=150, version=1)