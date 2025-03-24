from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
from brain import *

def main():
    no_neurons = 1000
    cap_size = 100
    area = Area(no_classes=10, cap_size=cap_size, n=no_neurons, in_n=784)
    (X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
    
    no_data_items = 70
    labels = y_train[:no_data_items]
    idxs = np.argsort(labels)
    X_train = [X_train[i] for i in idxs]        
    labels = [labels[i] for i in idxs]
    no_of_rounds = []
    for i in range(10):
        no_of_rounds.append(labels.count(i))
    
    input = np.array([np.matrix.flatten(x) for x in X_train])
    y, activations = area.training(input, no_rounds=no_of_rounds)
    

    diff = []
    for i in range(100):     
        out = area.predict(np.matrix.flatten(X_test[i]))
        expected = y_test[i]
        
        diff.append(out - expected)
        
        print("Expected: ", expected)
        print("Predicted: ", out)
        print("----------------------------------------")
    
    print("Accuracy: ", diff.count(0) / len(diff))
    
    X = np.array(list(y.values()))
    plt.imshow(X, cmap='coolwarm', interpolation='nearest')
    plt.savefig('assemblies.png')
    
    plt.imshow(area.weights, cmap='hot', interpolation='nearest')
    plt.savefig('weights.png')


    # setup for plotting
    idx = np.full(no_neurons, -1, dtype=int)
    
    act = activations[:, -1].copy()
    for i, j in enumerate(range(10)):
        idx[i*cap_size:(i+1)*cap_size] = act[j].argsort()[-cap_size:][::-1]
        act[:, idx[i*cap_size:(i+1)*cap_size]] = -1
    r = np.arange(no_neurons)
    r[idx[idx > -1]] = -1
    idx[(i+1)*cap_size:] = np.unique(r)[1:]


    outputs = np.zeros((10, 5, 784, no_neurons))
    for i in np.arange(10):
        for j in range(5):
            outputs[i, j] = activations[i,j]
    # Plot of firing neurons
    fig, axes = plt.subplots(10, 5, figsize=(10, 2 * 10), sharex=True, sharey=True)
    for ax, output in zip(axes, outputs):
        for i in range(5):
            ax[i].imshow((output[i] > 0)[:no_neurons, idx])
            ax[i].set_axis_off()
    fig.text(0.5, 0.04, 'Neurons', ha='center', va='center')
    fig.text(0.04, 0.5, 'Samples', ha='center', va='center', rotation='vertical')
    fig.savefig('firing_neurons.png')

    #Plot of firing probs
    # fig, ax = plt.subplots(figsize=(10, 4))
    # for i in range(10):
    #     ax.bar(np.arange(area.n), 1, label=i)
    # ax.legend(loc='upper right', ncol=2)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_ylim([0, 1.1])
    # ax.set_xticklabels([])
    # ax.set_xlabel('Neurons')
    # ax.set_ylabel('Firing Probability')
    # fig.savefig('firing_probs.png')
    

if __name__ == '__main__':
    main()