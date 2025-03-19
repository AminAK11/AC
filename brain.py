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
    
    def __init__(self, neurons = [], inhibited = False, p = 0.1, n = 100, cap_size = 5, beta= 0.1, p_r = 0.9, p_q = 0.1):
        self.neurons = neurons
        self.inhibited = inhibited
        self.weights = np.ones((len(neurons), len(neurons)))
        self.n = n
        self.p = p
        self.cap_size = cap_size
        self.beta = beta
        self.p_r = p_r
        self.p_q = p_q * self.cap_size / self.n

        

        # 0 - no connection, 1 - connection
        # Establishes initial connections betwen neurons in the brain area with prob p
        self.weights = np.random.choice([0, 1], (len(neurons), len(neurons)), p = [1 - p, p])
        np.fill_diagonal(self.weights, 0)

    def k_cap(self, SI, cap_size):
        out = np.zeros(len(SI))
        k_largest_index = np.argsort(SI)[-cap_size:]
        for i in range(len(SI)):
            if i in k_largest_index:
                out[i] = 1
        return out

    def training(self):
        no_classes = 2
        no_rounds = 2
        
        test_class_A = [np.concat([np.ones(self.cap_size,dtype=int), np.zeros(self.cap_size,dtype=int)]) for _ in range(no_classes // 2)]
        test_class_B = [np.concat([np.zeros(self.cap_size,dtype=int), np.ones(self.cap_size,dtype=int)]) for _ in range(no_classes // 2)]
        test_input = np.concat([test_class_A, test_class_B], dtype=int)

        for i in range (no_classes):
            n = len(self.neurons)
            in_class_activations = np.array([0] * n)
            for j in range (no_rounds):
                props = np.array([self.p_q] * self.cap_size * no_classes)
                props[j * self.cap_size : (j + 1) * self.cap_size] = self.p_r * np.ones(self.cap_size)
                props /= np.sum(props)
            
                
                in_class_activations = np.random.choice(
                    test_input[j],
                    p = props
                )

                SI = in_class_activations @ self.weights
                activations_t_1 = self.k_cap(SI, self.cap_size)
                self.weights = self.weights * (1 + self.beta * activations_t_1 * in_class_activations)
                in_class_activations = activations_t_1
              

        

    def inhibit(self):
        self.inhibited = True
    
    def step(self):
        if self.inhibited:
            return
        
        for i in range(len(self.neurons)):
            node = self.neurons[i]
            incoming_edges = self.weights[:,i]

            node.SI = np.sum([
                self.neurons[incoming_edge].fires() * self.weights[incoming_edge, i]
                for incoming_edge in incoming_edges
            ])

        fires_arr = [False for _ in range(self.neurons)]
        for n in self.neurons.sort(key = lambda x: x.SI)[self.cap_size:]:
            i = self.neurons.index(n)
            fires_arr[i] = True
            
        for i in range(len(self.neurons)):
            for j in range(len(self.neurons)):
                neuron_j = self.neurons[j]
                
                firesi_t_plus_1 = fires_arr[i]
                firesj_t = neuron_j.fires()
                
                self.weights[j, i] = self.weights[j, i] * (1 + self.beta * firesj_t * firesi_t_plus_1)
        
        for b, i in enumerate(fires_arr):
            self.neurons[i].fires = b



class Neuron():
    SI: int
    fires: bool
    
    def __init__(self):
        self.SI = 1
        self.fires = False

if __name__ == '__main__':
    area = Area(neurons = [Neuron() for i in range(20)])
    
    area.training()
    
