import unittest
import torch
from mnist_cnn import EfficientMNISTCNN, train_model
from unittest.case import TestCase
import os
import platform
from functools import wraps
import time
import torchvision
import torchvision.transforms.v2 as transforms
import numpy as np

class TestMNISTCNN(unittest.TestCase):
    def setUp(self):
        self.model = EfficientMNISTCNN()
        
    def test_parameter_count(self):
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertLess(param_count, 25000, 
            f"Model has {param_count} parameters, which exceeds the limit of 25,000")
        print(f"\nParameter count test passed. Total parameters: {param_count}")

    def test_readme_exists(self):
        """Test that README.md exists and is not empty."""
        self.assertTrue(os.path.exists('README.md'), "README.md file does not exist")
        
        with open('README.md', 'r') as f:
            content = f.read()
        
        self.assertGreater(len(content), 100, 
            "README.md seems too short. Should contain comprehensive documentation")
        print("\nREADME.md test passed. File exists and contains content")


    def test_model_accuracy(self):
        # Train the model and get accuracy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EfficientMNISTCNN().to(device)
        
        # Redirect stdout to capture prints during training
        import sys
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            train_model()
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # Extract accuracy from the output
        accuracy_line = [line for line in output.split('\n') if 'Test Accuracy' in line][0]
        accuracy = float(accuracy_line.split(':')[1].strip('%'))
        
        self.assertGreater(accuracy, 99.4, 
            f"Model accuracy {accuracy:.2f}% is below the required 99.4%")
        print(f"\nAccuracy test passed. Model achieved {accuracy:.2f}% accuracy")


if __name__ == '__main__':
    unittest.main()
