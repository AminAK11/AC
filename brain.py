import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import unittest

class Area():
    neurons: list
    inhibited: bool
    weights: np.array
    cap_size: int
    beta: float
    n: int
    p: float
    p_r: float
    p_q: float
    no_classes: int
    
    def __init__(self, p = 0.1, cap_size = 5, beta= 0.1, p_r = 0.9, p_q = 0.1, no_classes = 10, n = 100):
        self.no_classes = no_classes
        self.cap_size = cap_size
        self.n = n
        self.input_size = self.no_classes * self.cap_size
        self.weights = np.ones((self.input_size, n))
        self.p = p
        self.beta = beta
        self.p_r = p_r
        self.p_q = p_q * self.cap_size / self.n
        self.assembly_history = np.zeros((no_classes, self.n))
        self.y: dict[list[int], list[int]] = {}

        # 0 - no connection, 1 - connection
        # Establishes initial connections betwen neurons in the brain area with prob p
        self.weights = np.random.choice([0., 1.], (self.input_size, self.n), p = [1 - p, p])
        np.fill_diagonal(self.weights, 0)
        
    def k_cap(self, SI, cap_size):
        # Sorts the cap_size largest values in SI and returns their index
        out = np.zeros(len(SI))
        k_largest_index = np.argsort(SI)[-cap_size:]
        # returns an array of length n where the neurons with the k largest values are 1 and the rest are 0
        for i in range(len(SI)):
            if i in k_largest_index:
                out[i] = 1
        return out

    def test_classes(self, no_classes):
        n = no_classes * self.cap_size
        test_input = np.zeros((no_classes, n), dtype=int)
        
        for i in range(no_classes):
            test_input[i, i * self.cap_size :(i+1) * self.cap_size:] = 1
        
        return test_input

    def _get_propability_matrix(self, input):
        props = np.zeros((self.no_classes, self.input_size))
        for i in range(self.no_classes):
            for j in range(self.input_size):
                if (input[i, j] == 1):
                    props[i, j] = self.p_r
                else:
                    props[i, j] = self.p_q
        return props

    def _get_activations(self, in_class_activations):        
        SI = in_class_activations @ self.weights
        activations_t_1 = self.k_cap(SI, self.cap_size)
        return activations_t_1

    def training(self, input = None):
        no_rounds = 10
        
        test_input = input if input else self.test_classes(self.no_classes)

        props = self._get_propability_matrix(test_input)

        for i in range (self.no_classes):
            ps = props[i]
            for j in range (no_rounds):
                in_class_activations = np.array([1 if np.random.rand() < ps[x] else 0 for x in range(self.input_size)])
                activations_t_1 = self._get_activations(in_class_activations)
                self.assembly_history[i] += activations_t_1

                # Matrix of same size as weights 1 where in_class_activations is 1 and activations_t_1 is 1
                outer_prod = np.outer(in_class_activations, activations_t_1) * self.beta
                self.weights *= np.ones((len(in_class_activations), self.n)) + outer_prod
                
            self.y[i] = activations_t_1

        with open ("output.txt", "w") as f:
            f.write(f"test class \n {test_input}\n")
            for key, value in self.y.items():
                f.write(f"{key}: {value}\n")
        
        return self.y
    
    def predict(self, input):
        # From the output y find the likely class
        activations_t_1 = self._get_activations(input)
        
        smallest_dist = float('inf')
        likely_class = None
        for key, values in self.y.items():
            dist = np.sqrt(np.sum([(value - activation) ** 2 for value in values for activation in activations_t_1]))
            
            if dist < smallest_dist:
                smallest_dist = dist
                likely_class = key

        return likely_class
    
class Tests(unittest.TestCase):
    def test_predict_simple(self):
        pass

if __name__ == '__main__':
    area = Area()

    y = area.training()
    plt.imshow(area.weights, cmap='hot', interpolation='nearest')
    plt.savefig('weights.png')
    
    X = np.array(list(y.values()))
    Y = np.array(list(y.keys()))
    
    plt.imshow(X, cmap='coolwarm', interpolation='nearest')
    plt.savefig('assemblies.png')
    
    
    plt.imshow(area.assembly_history, cmap='coolwarm', interpolation='nearest')
    plt.savefig('assembly_history.png')
    
    input = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    test = [0 for _ in range(area.cap_size * (area.no_classes - 1))]
    test.extend([1 for _ in range(area.cap_size)])
    
    print(test)
    
    prediction = area.predict(test)
    print(prediction)
    
