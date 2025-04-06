import unittest
from brain import *
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def test_classes(area, no_classes):
        n = no_classes * area.cap_size
        test_input = np.zeros((no_classes, n), dtype=int)
        
        for j in range(no_classes):
            test_input[j, j * area.cap_size :(j+1) * area.cap_size:] = 1

        return test_input

class Tests(unittest.TestCase):
    def setUp(self):
        self.area = Area(no_classes=2, cap_size=140, n=200, in_n = 280)
        self.data = test_classes(self.area, self.area.no_classes)
        self.data_per_class = 20
        self.data = np.repeat(self.data, self.data_per_class, axis=0)
                
        self.acti = lambda x : (
            (x * np.random.choice([1, 0], size=x.shape, p=[self.area.p_r, 1-self.area.p_r])) |
            np.where(x != 1, 1, 0) * np.random.choice([1, 0], size=x.shape, p=[self.area.p_q, 1-self.area.p_q])
        )

        self.area.training(self.data, no_rounds=[self.data_per_class] * self.area.no_classes, activations_callback=self.acti)
        
        return super().setUp()
    
    def test_predict_simple(self):
        a = 0
        correct = 0
        for i, traning_data in enumerate(self.data):
            likely_class, _ = self.area.predict(traning_data, activations_callback=self.acti)
            
            a += 1 if i % self.data_per_class == 0 and i != 0 else 0
            t = i * self.data_per_class % self.area.no_classes + a
            
            correct += 1 if t == likely_class else 0
        
        accuracy = correct / len(self.data)
        print("Accuracy: ", accuracy)
        self.assertTrue(accuracy > 0.8)
        
    def test_plots(self):
        fig, ax = plt.subplots()
        def overlap(g, h):
            v = np.sum([a == 1 and b == 1 for (a, b) in zip(self.area.y[g], self.area.y[h])])
            return v / self.area.cap_size
        overlap_plot_matrix = np.zeros((self.area.no_classes, self.area.no_classes))
        for g in range(self.area.no_classes):
            for h in range(self.area.no_classes):
                overlap_plot_matrix[g,h] = overlap(g,h)
        
        overlap_plot_matrix /= overlap_plot_matrix.sum(axis = 0)
        for c in range(self.area.no_classes):
            overlap_plot_matrix[c,c] = 1 - sum(np.concatenate([overlap_plot_matrix[c, :c], overlap_plot_matrix[c, c+1:]]))
            
        for i in range(self.area.no_classes):
            for j in range(self.area.no_classes):
                ax.text(i,j, round(overlap_plot_matrix[i,j], self.area.no_classes), va='center', ha='center')
        
        ax.imshow(overlap_plot_matrix, cmap='viridis')
        ax.set_title('Overlap between two classes')
        fig.savefig('plots/overlaptwo.png')
        
        x = self.area.y[0]
        y = self.area.y[1]
        
        x_matrix = np.array(x).reshape(10, 20)
        y_matrix = np.array(y).reshape(10, 20)

        plt.figure()
        plt.scatter(*np.where(x_matrix == 1), color='blue', label='x=1', s=100, edgecolors='k', marker='s', alpha=0.5)
        plt.scatter(*np.where(y_matrix == 1), color='green', label='y=1', s=100, edgecolors='k', marker='s', alpha=0.5)
        plt.savefig('plots/overlapthree.png')

if __name__ == '__main__':
    unittest.main()
    # plots()