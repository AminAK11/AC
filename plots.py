import numpy as np
import matplotlib.pyplot as plt
from brain import *
from tensorflow import keras

def assemblies_and_weights():
    area = Area(no_classes=10, cap_size=250, n=2500, in_n=784, beta=0.1)
    (X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
    
    no_data_items = 100
    labels = y_train[:no_data_items]
    idxs = np.argsort(labels)
    X_train = [X_train[i] for i in idxs]        
    labels = [labels[i] for i in idxs]
    no_of_rounds = []
    for i in range(10):
        no_of_rounds.append(labels.count(i))
    
    print("no_of_rounds: ", no_of_rounds)
    
    input = np.array([np.matrix.flatten(x) for x in X_train])
    training_y = area.training(input, no_rounds=no_of_rounds)
    
    # Plotting the assemblies for each class
    res = []
    for y in training_y.values():
        res.append(np.array(y).reshape(50, 50))
    
    fig, axes = plt.subplots(2, len(res), figsize=(5 * len(res), 5))
    for i, y in enumerate(res):
        axes[0,i].imshow(res[i], cmap='coolwarm')
        axes[0,i].title.set_text(f'Assembly for number: {i}')

    test_index = 15
    number = y_test[test_index]
    data = X_test[test_index]
    predicted_class, assembly = area.predict(np.matrix.flatten(data))
    assembly = assembly.reshape(50, 50)
    for ax in axes[1]:
        ax.axis('off')
    axes[1, predicted_class].imshow(assembly, cmap='coolwarm')
    axes[1, predicted_class].title.set_text(f'Number: {number}, Predicted: {predicted_class}')
    plt.subplots_adjust(hspace=0.5)
    
    fig.savefig('plots/assemblies.png')

    # Plotting the weights
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(area.weights, cmap='hot', interpolation='nearest')
    fig.savefig('plots/weights.png')

def accuracy():
    neurons = []
    accuracies = []

    for i in range(1, 11):
        area = Area(no_classes=10, cap_size=250, n=1000*i, in_n=784, beta=0.1)
        (X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
        input = np.array([np.matrix.flatten(x) for x in X_train])
        input_test = np.array([x.flatten() for x in X_test])
        area.training(input, no_rounds=[6000]*10)
        acc = area.score(input_test, y_test)
        print(f"Accuracy for {i} neurons: {acc}")
        
        neurons.append(1000 * i)
        accuracies.append(acc)
        
    plt.plot(neurons, accuracies, 'ro-')
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of neurons')
    plt.grid(True)
    plt.savefig('plots/accuracy.png')
    plt.show()



if __name__ == '__main__':
    accuracy()
    assemblies_and_weights()