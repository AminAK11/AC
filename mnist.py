from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
from brain import *

def main():
    brain = Brain(no_classes=10, cap_size=100, n=1000, in_n=784, beta=0.1)
    (X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
    
    no_data_items = 1000
    labels = y_train[:no_data_items]
    idxs = np.argsort(labels)
    X_train = [X_train[i] for i in idxs]        
    labels = [labels[i] for i in idxs]
    no_of_rounds = []
    for i in range(10):
        no_of_rounds.append(labels.count(i))
    
    print("no_of_rounds: ", no_of_rounds)
    
    input = np.array([np.matrix.flatten(x) for x in X_train])
    y = brain.training(input, no_rounds=no_of_rounds)
    
    
    diff = []
    for i in range(100):     
        likely_class, activations_t_1 = brain.predict(np.matrix.flatten(X_test[i]))
        expected = y_test[i]
        
        
        diff.append(likely_class - expected if likely_class > expected else expected - likely_class)

        
        if i >= 10: continue
        print("Expected: ", expected)
        print("Predicted: ", likely_class)
        print("----------------------------------------")
    
    print("Accuracy: ", diff.count(0) / len(diff))
    
    
    # X = np.array(list(y.values()))
    # plt.imshow(X, cmap='coolwarm', interpolation='nearest')
    # plt.savefig('assemblies.png')
    
    # plt.imshow(brain.weights, cmap='hot', interpolation='nearest')
    # plt.savefig('weights.png')

    # Plot of firing probs
    # fig, ax = plt.subplots(figsize=(10, 4))
    # for i in range(10):
    #     ax.bar(np.arange(brain.n), y[i, -1].mean(axis=0)[idx], label=i)
    # ax.legend(loc='upper right', ncol=2)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_ylim([0, 1.1])
    # ax.set_xticklabels([])
    # ax.set_xlabel('Neurons')
    # ax.set_ylabel('Firing Probability')
    

if __name__ == '__main__':
    main()