import numpy as np
import matplotlib.pyplot as plt
from brain import *

from tensorflow import keras
import time

def get_data(no_data_items, X_train, y_train):
    labels = y_train[:no_data_items]
    idxs = np.argsort(labels)
    X_train = [X_train[i] for i in idxs]
    labels = [labels[i] for i in idxs]

    no_of_rounds = [labels.count(z) for z in range(10)]
    lowest_data_number = np.amin(no_of_rounds)
    
    data = []
    new_labels = []
    class_counts = {i: 0 for i in range(10)}

    for x, y in zip(X_train, labels):
        if class_counts[y] < lowest_data_number:
            data.append(x)
            new_labels.append(y)
            class_counts[y] += 1

        if all(count == lowest_data_number for count in class_counts.values()):
            break

    no_of_rounds = [lowest_data_number] * 10
    
    return np.array([x.flatten() for x in data]), new_labels, no_of_rounds

def assemblies_and_weights(cap_size=200, beta=1):
    brain = Brain(no_classes=10, cap_size=cap_size, n=5625, in_n=784*4, beta=beta)
    (X_train, y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
    data, labels, no_of_rounds = get_data(300, X_train, y_train)
    assembly_size = 75
    no_test_data = 1000
    
    training_y = brain.section_training(data, no_rounds=no_of_rounds)
    score = brain.score(data[:no_test_data], labels[:no_test_data])
    
    test_data = np.array([x.flatten() for x in X_test])
    test_score = brain.score(test_data[:no_test_data], y_test[:no_test_data])
    
    print("Training accuracy: ", score)
    print("Test training accuracy: ", test_score)

    # Plotting the assemblies for each class
    res = []
    for y in training_y.values():
        res.append(np.array(y).reshape(assembly_size, assembly_size))
    
    fig, axes = plt.subplots(4, len(res), figsize=(5 * len(res), 5))
    for i, y in enumerate(res):
        axes[0,i].imshow(res[i], cmap='viridis', interpolation='nearest')
        axes[0,i].title.set_text(f'Assembly for number: {i}')

    for ax1, ax2, ax3 in zip(axes[1], axes[2], axes[3]):
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
    
    for i in range(45):
        test_index = 13 * i
        number = y_test[test_index]
        data = X_test[test_index]
        
        predicted_class, assembly = brain.predict(data.flatten())
        assembly = assembly.reshape(assembly_size, assembly_size)

        acti = brain._get_in_class_activations(data.flatten()).reshape(56, 56)
        axes[3, predicted_class].imshow(data, cmap='viridis', interpolation='nearest')
        axes[2, predicted_class].imshow(acti.reshape(56, 56), cmap='viridis', interpolation='nearest')
        axes[1, predicted_class].imshow(assembly, cmap='viridis', interpolation='nearest')
        axes[1, predicted_class].title.set_text(f'Number: {number}, Predicted: {predicted_class}')
    
    plt.subplots_adjust(hspace=0.8)
    fig.savefig('plots/assemblies.png')

    # Plotting the weights
    fig, ax = plt.subplots(figsize=(75, 75))
    ax.imshow(brain.weights, cmap='hot', interpolation='nearest')
    fig.savefig('plots/weights.png')


def benchmark_param(
    brain_callback,
    item_callback,
    name,
    no_data_items=1000,
    no_test_data=500,
    no_iterations=10,
):
    items = []
    test_accuracies = []
    train_accuracy = []
    
  
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()    
    y_train = y_train[:no_data_items]
    
    idxs = np.argsort(y_train)
    X_train = [X_train[j] for j in idxs]
    y_train = [y_train[k] for k in idxs]
    no_of_rounds = [y_train.count(z) for z in range(10)]
    
    input = np.array([x.flatten() for x in X_train])
    input_test = np.array([p.flatten() for p in X_test])
    
    '''
    Accuracy before training
    '''
    brain = brain_callback(0)
    brain.section_training(input, no_rounds=[1]*10)
    first_test_acc = brain.score(input_test[:no_test_data], y_test[:no_test_data])
    first_train_acc = brain.score(input, y_train)
    
    items.append(item_callback(0))
    test_accuracies.append(first_test_acc)
    train_accuracy.append(first_train_acc)

    for i in range(1, no_iterations + 1):
        # y_train[:no_of_rounds[0]], y_train[no_of_rounds[0]:no_of_rounds[1]] = y_train[no_of_rounds[-1]], y_train[:no_of_rounds[0]]
        # print("no_of_rounds", no_of_rounds)
        
        # y_train[:no_of_rounds[0]], y_train[np.sum(no_of_rounds[:-1]):] = y_train[np.sum(no_of_rounds[:-1]):], y_train[:no_of_rounds[0]]
        
        # X_train[:no_of_rounds[0]], X_train[np.sum(no_of_rounds[:-1]):] = X_train[np.sum(no_of_rounds[:-1]):], X_train[:no_of_rounds[0]]
        
        # no_of_rounds[0], no_of_rounds[-1] = no_of_rounds[-1], no_of_rounds[0]

        brain = brain_callback(i)
        brain.section_training(input, no_rounds=no_of_rounds)
        
        train_acc = brain.score(input[:no_data_items], y_train[:no_data_items])
        acc = brain.score(input_test[:no_test_data], y_test[:no_test_data])

        print(f"Accuracy for {name} {item_callback(i)}: {acc}")
        items.append(item_callback(i))
        test_accuracies.append(acc)
        train_accuracy.append(train_acc)

    plt.figure()
    plt.plot(items, test_accuracies, 'ro-')
    plt.plot(items, train_accuracy, 'ro-', color='blue')
    plt.xlabel(name)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs {name}')
    plt.grid(True)
    plt.savefig(f'plots/{name}1')
    
    return items[np.argmax(test_accuracies)]

def overlap_plots(brain, no_data_items=1000):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()    
    y_train = y_train[:no_data_items]
        
    idxs = np.argsort(y_train)
    X_train = [X_train[j] for j in idxs]
    y_train = [y_train[k] for k in idxs]
    no_of_rounds = [y_train.count(z) for z in range(10)]
    
    input = np.array([x.flatten() for x in X_train])
    input_test = np.array([p.flatten() for p in X_test])
    
    brain.section_training(input, no_rounds=no_of_rounds)
    brain.predict(input_test[0].flatten())
    
    fig2, ax = plt.subplots()
    def overlap(g, h):
        v = np.sum([a == 1 and b == 1 for (a, b) in zip(brain.y[g], brain.y[h])])
        return v / brain.cap_size
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

    ax.imshow(overlap_plot_matrix, cmap='viridis',)
    ax.set_title('Overlap between classes')
    fig2.savefig('plots/overlap.png')

if __name__ == '__main__':
    start = time.time()

    # best_cap_size = benchmark_param(
    #     brain_callback=lambda i: Brain(p=0.2, no_classes=10, cap_size=50*i + 50, n=500, in_n=784, beta=0.2),
    #     item_callback=lambda i: 50*i,
    #     name="cap_size",
    #     no_data_items=500,
    #     no_test_data=500,
    #     no_iterations=10
    # )

    # best_beta = benchmark_param(
    #     brain_callback=lambda i : Brain(no_classes=10, cap_size=100, n=1000, in_n=784, beta=0.1 * i),
    #     item_callback=lambda i: 0.1 * i,
    #     name="beta",
    #     no_data_items=200,
    #     no_test_data=50,
    # )

    # best_neuron_count = benchmark_param(
    #     brain_callback=lambda i: Brain(no_classes=10, cap_size=1000, n=1000 * i + 1000, in_n=784, beta=0.2),
    #     item_callback=lambda i: 1000 * i,
    #     name="number_of_neurons",
    #     no_data_items=10000,
    #     no_test_data=1000,
    # )
    
    # best_neuron_count = benchmark_param(
    #     brain_callback=lambda i: Brain(no_classes=10, cap_size=(i * 100 + 500) // 10, n=1000 * i + 1000, in_n=784, beta=0.1),
    #     item_callback=lambda i: 500 * i + 500,
    #     name="number_of_neurons",
    #     no_data_items=300,
    #     no_test_data=300,
    # )

    # best_neuron_count = benchmark_param(
    #     brain_callback=lambda i: Brain(p=0.1, no_classes=10, cap_size=10, n=1000, in_n=784, beta=0.1),
    #     item_callback=lambda i: 100,
    #     name="number_of_neurons",
    #     no_data_items=100,
    #     no_test_data=100,
    #     no_iterations=1
    # )
    
    # overlap_plots(
    #     Brain(p=0.1, no_classes=10, cap_size=80, n=200, in_n=784, beta=0.1),
    #     no_data_items=1000
    # )

    assemblies_and_weights()
    
    partial_time = time.time() - start
    print(f"Time taken: {partial_time}")