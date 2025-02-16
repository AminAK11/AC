import numpy as np

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
    k_cap: int
    beta: float
    n: int
    p: float
    
    def __init__(self, neurons = [], inhibited = False, p = 0.1, n = 100, k_cap = 10, beta= 0.1):
        self.neurons = neurons
        self.inhibited = inhibited
        self.weights = np.ones((len(neurons), len(neurons)))
        self.n = n
        self.p = p
        self.k_cap = k_cap
        self.beta = beta
                
        # 0 - no connection, 1 - connection
        self.weights = np.random.choice([0, 1], (len(neurons), len(neurons)), p = [1 - p, p])
        
        
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
        for n in self.neurons.sort(key = lambda x: x.SI)[self.k_cap:]:
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
    area = Area(neurons = [Neuron() for i in range(100)])