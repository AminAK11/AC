import numpy as np
import matplotlib.pyplot as plt
from brain import *

from tensorflow import keras
import time

def assemblies_and_weights(cap_size=1000, beta=0.1):
    area = Area(no_classes=10, cap_size=cap_size, n=10000, in_n=784, beta=beta)
    (X_train, y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
    
    labels = y_train
    idxs = np.argsort(labels)
    X_train = [X_train[i] for i in idxs]        
    labels = [labels[i] for i in idxs]

    input = np.array([x.flatten for x in X_train])
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
    predicted_class, assembly = area.predict(data.flatten())
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


def benchmark_param(
    area_callback,
    item_callback,
    name,
    no_data_items=1000,
    no_test_data=500,
    no_iterations=10,
):
    items = []
    test_accuracies = []
    train_accuracy = []
    
    area = area_callback(0)
    # (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # X_train = np.array([x.flatten() for x in X_train])
    # y_train = np.array([x.flatten() for x in y_train])
    # X_test = np.array([x.flatten() for x in X_test])
    # y_test = np.array([x.flatten() for x in y_test])
    # first_test_acc = area.score(X_test[:no_test_data], y_test[:no_test_data])
    # first_train_acc = area.score(X_train[:no_test_data], y_train[:no_test_data])

    # items.append(item_callback(0))
    # test_accuracies.append(first_test_acc)
    # train_accuracy.append(first_train_acc)

    for i in range(1, no_iterations + 1):
        area = area_callback(i)
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        y_train = y_train[:no_data_items]
        
        
        idxs = np.argsort(y_train)
        X_train = [X_train[j] for j in idxs]
        y_train = [y_train[j] for j in idxs]
        

        no_of_rounds = [y_train.count(z) for z in range(10)]
        
        # y_train[:no_of_rounds[0]], y_train[no_of_rounds[0]:no_of_rounds[1]] = y_train[no_of_rounds[-1]], y_train[:no_of_rounds[0]]
        #print("no_of_rounds", no_of_rounds)
        
        # y_train[:no_of_rounds[0]], y_train[np.sum(no_of_rounds[:-1]):] = y_train[np.sum(no_of_rounds[:-1]):], y_train[:no_of_rounds[0]]
        
        # X_train[:no_of_rounds[0]], X_train[np.sum(no_of_rounds[:-1]):] = X_train[np.sum(no_of_rounds[:-1]):], X_train[:no_of_rounds[0]]
        
        #no_of_rounds[0], no_of_rounds[-1] = no_of_rounds[-1], no_of_rounds[0]

        input = np.array([x.flatten() for x in X_train])
        input_test = np.array([x.flatten() for x in X_test])

        y = area.training(input, no_rounds=no_of_rounds)
        
        train_acc = area.score(input[:no_data_items], y_train[:no_data_items])
        acc = area.score(input_test[:no_test_data], y_test[:no_test_data])

        print(f"Accuracy for {name} {item_callback(i)}: {acc}")
        items.append(item_callback(i))
        test_accuracies.append(acc)
        train_accuracy.append(train_acc)

    fig2, ax = plt.subplots()
    def overlap(g, h):
        v = np.sum([a == 1 and b == 1 for (a, b) in zip(y[g], y[h])])
        return v / len(y[g])
    overlap_plot_matrix = np.zeros((10, 10))
    for g in range(10):
        for h in range(10):
            overlap_plot_matrix[g,h] = overlap(g,h)
    
    overlap_plot_matrix /= overlap_plot_matrix.sum(axis = 0)
    for c in range(10):
        overlap_plot_matrix[c,c] = 1 - sum(np.concatenate([overlap_plot_matrix[c, :c], overlap_plot_matrix[c, c+1:]]))
        
    for i in range(10):
        for j in range(10):
            ax.text(i,j, round(overlap_plot_matrix[i,j], 2), va='center', ha='center')

    ax.imshow(overlap_plot_matrix, cmap='cool',)
    ax.set_title('Overlap between classes')
    fig2.savefig('plots/overlap1.png')

    plt.figure()
    plt.plot(items, test_accuracies, 'ro-')
    plt.plot(items, train_accuracy, 'ro-', color='blue')
    plt.xlabel(name)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs {name}')
    plt.grid(True)
    plt.savefig(f'plots/{name}1')
    
    return items[np.argmax(test_accuracies)]

if __name__ == '__main__':
    start = time.time()

    # Running benchmarks with lambdas directly inline
    # best_cap_size = benchmark_param(
    #     area_callback=lambda i: Area(no_classes=10, cap_size=250 * i, n=10000, in_n=784, beta=0.1),
    #     item_callback=lambda i: 250 * i,
    #     name="cap_size",
    #     no_data_items=200,
    #     no_test_data=50
    # )

    # best_beta = benchmark_param(
    #     area_callback=lambda i, best_cap_size=best_cap_size: Area(no_classes=10, cap_size=best_cap_size, n=10000, in_n=784, beta=0.1 * i),
    #     item_callback=lambda i: 0.1 * i,
    #     name="beta",
    #     no_data_items=200,
    #     no_test_data=50,
    #     best_cap_size=best_cap_size
    # )

    # best_neuron_count = benchmark_param(
    #     area_callback=lambda i: Area(no_classes=10, cap_size=1000, n=1000 * i + 1000, in_n=784, beta=0.2),
    #     item_callback=lambda i: 1000 * i,
    #     name="number_of_neurons",
    #     no_data_items=10000,
    #     no_test_data=1000,
    # )
    
    # best_neuron_count = benchmark_param(
    #     area_callback=lambda i: Area(no_classes=10, cap_size=(i * 100 + 500) // 10, n=1000 * i + 1000, in_n=784, beta=0.1),
    #     item_callback=lambda i: 500 * i + 500,
    #     name="number_of_neurons",
    #     no_data_items=300,
    #     no_test_data=300,
    # )

    best_neuron_count = benchmark_param(
        area_callback=lambda i: Area(no_classes=10, cap_size=30, n=100, in_n=784, beta=0.1),
        item_callback=lambda i: 100,
        name="number_of_neurons",
        no_data_items=100,
        no_test_data=100,
        no_iterations=1
    )
    

    # assemblies_and_weights(cap_size=1000, beta=0.1)
    
    partial_time = time.time() - start
    print(f"Final result: \n Cap size: {"best_cap_size"},\n Beta: {"best_beta"},\n Number of neurons: {best_neuron_count}")
    print(f"Time taken: {partial_time}")