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
    
    def __init__(self, p = 0.1, cap_size = 10, beta=0.1, p_r = 0.9, p_q = 0.1, no_classes = 10, n = 100, in_n = 784):
        self.no_classes = no_classes
        self.cap_size = cap_size
        self.n = n
        self.input_size = in_n
        self.weights = np.ones((self.input_size, n))
        self.p = p
        self.beta = beta
        self.p_r = p_r
        self.p_q = p_q * self.cap_size / self.n
        self.y: dict[int, list[int]] = {}

        # 0 - no connection, 1 - connection
        # Establishes initial connections betwen neurons in the brain area with prob p
        self.weights = np.random.choice([0., 1.], (self.input_size, self.n), p = [1 - p, p])
        self.weights /= self.weights.sum(axis=0)
        
    def _k_cap(self, arr, cap_size):
        """ Sorts the cap_size largest values in the input array and returns their indices.
        Args:
            arr (np.ndarray): The array to be k-capped.
            cap_size (int): The number of elements to be selected.
        Returns: 
            np.ndarray: An array of length n where the neurons with the k largest values are 1 and the rest are 0.
        """
        out = np.zeros(len(arr))
        k_largest_index = np.argsort(arr)[-cap_size:]
        for i in range(len(arr)):
            if i in k_largest_index:
                out[i] = 1
        return out

    def _get_activations(self, in_class_activations, bias=None):
        SI = in_class_activations @ self.weights + (bias if bias is not None else 0)
        activations_t_1 = self._k_cap(SI, self.cap_size)
        return activations_t_1
    
    def _get_in_class_activations(self, input, activations_callback=None):
        if activations_callback is not None: return activations_callback(input)
        
        normed_input = input / 255
        out = np.astype(np.random.uniform(0, 1, (self.input_size)) < normed_input, np.int64)
        return out

    def training(self, test_input = None, no_rounds = None, activations_callback=None):
        #bias = np.zeros(self.n)
        #bias_penalty = -1
        
        for i in range(self.no_classes):
            print(f"Training class {i} using {self.n} neurons and {no_rounds[i]} rounds")
            no_input = sum(no_rounds[:i])
            for j in range(no_rounds[i]):
                in_class_activations = self._get_in_class_activations(test_input[no_input + j], activations_callback)
                activations_t_1 = self._get_activations(in_class_activations, None)
                outer_prod = np.outer(in_class_activations, activations_t_1) * self.beta
                self.weights = self.weights * (np.ones((len(in_class_activations), self.n)) + outer_prod)


            self.y[i] = np.where(activations_t_1 > 0, 1, 0)
            self.weights /= self.weights.sum(axis=0)
            #bias[activations_t_1 > 0] += bias_penalty
    
        with open ("output.txt", "w") as f:
            f.write(f"test class \\n")
            for test in test_input:
                f.write(f"{test}\n")

            for key, value in self.y.items():
                f.write(f"{key}: {value}\n")
        
        return self.y

    def predict(self, input, activations_callback=None):
        """Predicts the class of the input.
            
        Returns:
            Most likely class. 
        """
        activations_t_1 = self._get_activations(self._get_in_class_activations(input, activations_callback))
        
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
        
        wrong = {i: 0 for i in range(self.no_classes)}
        for x, y in zip(X_test, y_test):

            likely_class, _ = self.predict(x)
            if likely_class == y:
                correct += 1
            else:
                wrong[y] += 1
            
        print("Wrong predictions:", wrong)
        return correct / len(X_test)
