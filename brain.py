import numpy as np
rng = np.random.default_rng()

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
    
    def __init__(self, p = 0.1, cap_size = 10, beta=0.1, p_r = 0.9, p_q = 0.1, no_classes = 10, n = 100, in_n = 100):
        self.no_classes = no_classes
        self.cap_size = cap_size
        self.n = n
        self.input_size = in_n
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

    def _get_activations(self, in_class_activations, bias=None):
        SI = in_class_activations @ self.weights + (bias if bias is not None else 0)

        activations_t_1 = self.k_cap(SI, self.cap_size)
        return activations_t_1

    def training(self, input = None, no_rounds = None):    
        test_input = input if input is not None else self.test_classes(self.no_classes)

        bias = np.zeros(self.n)
        bias_penalty = -1

        for i in range(self.no_classes):
            print(f"Training class {i} using {self.n} neurons and {no_rounds[i]} rounds")
            for _ in range(no_rounds[i]):
                highest = self.k_cap(test_input[sum(no_rounds[:i])], self.cap_size)
                prop = [self.p_r if h >= 1 else self.p_q for h in highest]
                
                in_class_activations = np.array([(1 if np.random.rand() < prop[x] else 0) for x in range(self.input_size)])
                activations_t_1 = self._get_activations(in_class_activations, bias)
                self.assembly_history[i] += activations_t_1
                
                # Matrix of same size as weights. Beta if both in_class_activations and activations_t_1 is 1
                outer_prod = np.outer(in_class_activations, activations_t_1) * self.beta
                self.weights = self.weights * (np.ones((len(in_class_activations), self.n)) + outer_prod)

            self.y[i] = activations_t_1
            self.weights /= self.weights.sum(axis=0)
            bias[activations_t_1 > 0] += bias_penalty
        
        return self.y
    
    def predict(self, input):
        """Predicts the class of the input.
            
        Returns:
            Most likely class. 
        """
        
        highest = self.k_cap(input, self.cap_size)
        prop = [self.p_r if h >= 1 else self.p_q for h in highest]
        in_class_activations = np.array([(1 if np.random.rand() < prop[x] else 0) for x in range(self.input_size)])
        activations_t_1 = self._get_activations(in_class_activations)
        
        likely_class = None
        best_score = 0
        
        for key, values in self.y.items():
            score = np.sum(np.array(values) == activations_t_1)
            if score > best_score:
                best_score = score
                likely_class = key

        return likely_class, activations_t_1

    def score(self, X_test, y_test):
        correct = 0
        for x, y in zip(X_test, y_test):
            likely_class, _ = self.predict(x)
            if likely_class == y:
                correct += 1

        return correct / len(X_test)


