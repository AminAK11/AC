import numpy as np
import matplotlib.pyplot as plt
from brain import *
from tensorflow import keras

def assemblies_and_weights(cap_size=200, beta=0.1):
    area = Area(no_classes=10, cap_size=cap_size, n=10000, in_n=784, beta=beta)
    (X_train, y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
    
    labels = y_train
    idxs = np.argsort(labels)
    X_train = [X_train[i] for i in idxs]        
    labels = [labels[i] for i in idxs]

    input = np.array([np.matrix.flatten(x) for x in X_train])
    training_y = area.training(input, no_rounds=[6000]*10)
    
    # Plotting the assemblies for each class
    res = []
    for y in training_y.values():
        res.append(np.array(y).reshape(100, 100))
    
    fig, axes = plt.subplots(2, len(res), figsize=(5 * len(res), 5))
    for i, y in enumerate(res):
        axes[0,i].imshow(res[i], cmap='binary')
        axes[0,i].title.set_text(f'Assembly for number: {i}')

    test_index = 15
    number = y_test[test_index]
    data = X_test[test_index]
    predicted_class, assembly = area.predict(np.matrix.flatten(data))
    assembly = assembly.reshape(100, 100)
    for ax in axes[1]:
        ax.axis('off')
    axes[1, predicted_class].imshow(assembly, cmap='binary')
    axes[1, predicted_class].title.set_text(f'Number: {number}, Predicted: {predicted_class}')
    plt.subplots_adjust(hspace=0.5)
    
    fig.savefig('plots/assemblies.png')

    # Plotting the weights
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(area.weights, cmap='hot', interpolation='nearest')
    fig.savefig('plots/weights.png')

def accuracy(cap_size=200, beta=0.1):
    neurons = []
    accuracies = []

    for i in range(1, 11):
        area = Area(no_classes=10, cap_size=cap_size, n=1000*i, in_n=784, beta=beta)
        (X_train, y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
        
        idxs = np.argsort(y_train)
        X_train = [X_train[j] for j in idxs]        
        y_train = [y_train[j] for j in idxs]
        
        input = np.array([np.matrix.flatten(x) for x in X_train])
        input_test = np.array([x.flatten() for x in X_test])
        
        area.training(input, no_rounds=[6000]*10)
        
        acc = area.score(input_test, y_test)
        print(f"Accuracy for {i}000 neurons: {acc}")
        
        neurons.append(100 * i)
        accuracies.append(acc)
        
    plt.plot(neurons, accuracies, 'ro-')
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of neurons')
    plt.grid(True)
    plt.savefig('plots/accuracy.png')
    plt.show()
        
def best_cap_size():
    cap_sizes = []
    accuracies = []

    for i in range(1, 11):
        area = Area(no_classes=10, cap_size=250*i, n=10000, in_n=784, beta=0.1)
        (X_train, y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
        
        idxs = np.argsort(y_train)
        X_train = [X_train[j] for j in idxs]        
        y_train = [y_train[j] for j in idxs]
        
        input = np.array([np.matrix.flatten(x) for x in X_train])
        input_test = np.array([x.flatten() for x in X_test])
            
        area.training(input, no_rounds=[6000]*10)
        acc = area.score(input_test, y_test)
        print(f"Accuracy for cap size {250*i}: {acc}")
        
        cap_sizes.append(250 * i)
        accuracies.append(acc)
        
    plt.plot(cap_sizes, accuracies, 'ro-')
    plt.xlabel('Cap size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Cap size')
    plt.grid(True)
    plt.savefig('plots/cap_size.png')
    plt.show()
    
    return cap_sizes[accuracies.index(max(accuracies)) + 1]

def best_beta(cap_size):
    betas = []
    accuracies = []

    for i in range(1, 11):
        area = Area(no_classes=10, cap_size=cap_size, n=10000, in_n=784, beta=0.1 * i)
        (X_train, y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
        input = np.array([np.matrix.flatten(x) for x in X_train])
        input_test = np.array([x.flatten() for x in X_test])

        y_train = y_train
        idxs = np.argsort(y_train)
        X_train = [X_train[j] for j in idxs]        
        y_train = [y_train[j] for j in idxs]
        
        area.training(input, no_of_rounds=[6000]*10)
        acc = area.score(input_test, y_test)
        print(f"Accuracy for beta {0.1*i}: {acc}")
        
        betas.append(0.1 * i)
        accuracies.append(acc)
        
    plt.plot(betas, accuracies, 'ro-')
    plt.xlabel('Beta')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Beta')
    plt.grid(True)
    plt.savefig('plots/beta.png')
    plt.show()
    
    return betas[accuracies.index(max(accuracies)) + 1]


if __name__ == '__main__':
    cap_size = best_cap_size()
    beta = best_beta(cap_size)
    accuracy(cap_size=cap_size, beta=beta)
    assemblies_and_weights(cap_size=cap_size, beta=beta)