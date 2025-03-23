import unittest
from brain import *
import numpy as np

class Tests(unittest.TestCase):
    def setUp(self):
        self.area = Area(no_classes=20, cap_size=60, n=500, in_n=1200)
        self.data = self.area.test_classes(self.area.no_classes)
        self.area.training(input=self.data)
        
        return super().setUp()
    
    def test_predict_simple(self):
        for i, traning_data in enumerate(self.data):           
            noise = i * 0.01
            
            r = np.random.choice([0, 1], size=self.data[0].shape, p=[noise, 1 - noise]) 
            self.assertEqual(i, self.area.predict(traning_data * r))
            

if __name__ == '__main__':
    unittest.main()