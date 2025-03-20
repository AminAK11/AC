import numpy as np
rng = np.random.default_rng()

class Brain():     
    areas: list
    n: int
    p: float
    
    def __init__(self, p = 0.1, n = 100):
        self.areas = []
        self.p = p
        self.n = n

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
    
    def __init__(self, neurons = [], inhibited = False, p = 0.1, cap_size = 5, beta= 0.1, p_r = 0.9, p_q = 0.1, no_classes = 4):
        self.neurons = neurons
        self.no_classes = no_classes
        self.cap_size = cap_size
        self.n = self.no_classes * self.cap_size
        self.inhibited = inhibited
        self.weights = np.ones((len(neurons), len(neurons)))
        self.p = p
        self.beta = beta
        self.p_r = p_r
        self.p_q = p_q * self.cap_size / self.n

        # 0 - no connection, 1 - connection
        # Establishes initial connections betwen neurons in the brain area with prob p
        self.weights = np.random.choice([0, 1], (self.n, self.n), p = [1 - p, p])
        np.fill_diagonal(self.weights, 0)

    def k_cap(self, SI, cap_size):
        out = np.zeros(len(SI))
        k_largest_index = np.argsort(SI)[-cap_size:]
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

    def training(self):
        no_rounds = 10
        
        test_input = self.test_classes(self.no_classes)
        
        y: dict[list[int], list[int]] = {}

        props = np.zeros((self.no_classes, self.n))
        for i in range(self.no_classes):
            for j in range(self.n):
                if (test_input[i, j] == 1):
                    props[i, j] = self.p_r
                else:
                    props[i, j] = self.p_q

        for i in range (self.no_classes):
            # props = np.array([self.p_q] * self.n)
            # props[i*self.cap_size:(i+1)*self.cap_size] = self.p_r * np.ones(self.cap_size)
            ps = props[i]
            for j in range (no_rounds):
                
                in_class_activations = np.array([1 if np.random.rand() < ps[i] else 0 for i in range(self.n)])
                
                SI = in_class_activations @ self.weights
                                
                activations_t_1 = self.k_cap(SI, self.cap_size)
                self.weights = self.weights * (1 + self.beta * activations_t_1 * in_class_activations)

            y[i] = activations_t_1
            
        print(y)
        print(self.weights)
        return y
    
    def inhibit(self):
        self.inhibited = True
    
    # def step(self):
    #     if self.inhibited:
    #         return
        
    #     for i in range(len(self.neurons)):
    #         node = self.neurons[i]
    #         incoming_edges = self.weights[:,i]

    #         node.SI = np.sum([
    #             self.neurons[incoming_edge].fires() * self.weights[incoming_edge, i]
    #             for incoming_edge in incoming_edges
    #         ])

    #     fires_arr = [False for _ in range(self.neurons)]
    #     for n in self.neurons.sort(key = lambda x: x.SI)[self.cap_size:]:
    #         i = self.neurons.index(n)
    #         fires_arr[i] = True
            
    #     for i in range(len(self.neurons)):
    #         for j in range(len(self.neurons)):
    #             neuron_j = self.neurons[j]
                
    #             firesi_t_plus_1 = fires_arr[i]
    #             firesj_t = neuron_j.fires()
                
    #             self.weights[j, i] = self.weights[j, i] * (1 + self.beta * firesj_t * firesi_t_plus_1)
        
    #     for b, i in enumerate(fires_arr):
    #         self.neurons[i].fires = b



class Neuron():
    SI: int
    fires: bool
    
    def __init__(self):
        self.SI = 1
        self.fires = False

if __name__ == '__main__':
    area = Area(neurons = [Neuron() for i in range(10)])
    
    area.training()
    
