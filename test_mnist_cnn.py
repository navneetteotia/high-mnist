import unittest
import torch
from mnist_cnn import LightweightMNISTCNN, train_model
from unittest.case import TestCase
import os
import platform
from functools import wraps
import time
import torchvision
import torchvision.transforms.v2 as transforms
import numpy as np

def timeout(seconds=10, error_message="Timeout"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if platform.system() == 'Windows':
                # Simple timeout implementation for Windows
                start_time = time.time()
                result = func(*args, **kwargs)
                if time.time() - start_time > seconds:
                    raise TimeoutError(error_message)
                return result
            else:
                # Unix-based systems can use SIGALRM
                import signal
                def _handle_timeout(signum, frame):
                    raise TimeoutError(error_message)

                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
        return wrapper
    return decorator

class TestMNISTCNN(unittest.TestCase):
    def setUp(self):
        self.model = LightweightMNISTCNN()
        
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

    def test_cnn_layer_count(self):
        """Test that the model has at least 2 CNN layers with proper channel dimensions."""
        # Get all modules in sequential order
        modules = list(self.model.features.children())
        
        # Check CNN layers count
        cnn_layers = [module for module in modules 
                     if isinstance(module, torch.nn.Conv2d)]
        
        self.assertGreaterEqual(len(cnn_layers), 2, 
            f"Model should have at least 2 CNN layers, but found {len(cnn_layers)}")
        
        # Check channel dimensions
        self.assertEqual(cnn_layers[0].in_channels, 1, 
            "First CNN layer should have 1 input channel for MNIST")
        self.assertEqual(cnn_layers[-1].out_channels, 12, 
            "Final CNN layer should have 12 output channels")
        
        print(f"\nCNN layer count test passed. Found {len(cnn_layers)} CNN layers")

    def test_maxpool_after_conv(self):
        """Test that each Conv2D layer is followed by MaxPool2d (allowing for ReLU)."""
        # Get all modules in sequential order
        modules = list(self.model.features.children())
        
        # Check for MaxPool after Conv2D
        for i, module in enumerate(modules[:-1]):  # Check all but last module
            if isinstance(module, torch.nn.Conv2d):
                next_module = modules[i + 1]
                self.assertTrue(
                    isinstance(next_module, torch.nn.MaxPool2d) or 
                    isinstance(modules[i + 2], torch.nn.MaxPool2d),  # Allow for ReLU in between
                    f"Conv2D layer should be followed by MaxPool2d (allowing for ReLU in between)"
                )
        
        print("\nMaxPool placement test passed. Each Conv2D layer is properly followed by MaxPool2d")

    @timeout(300)  # 5 minutes timeout
    def test_model_accuracy(self):
        # Train the model and get accuracy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LightweightMNISTCNN().to(device)
        
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
        
        self.assertGreater(accuracy, 95.0, 
            f"Model accuracy {accuracy:.2f}% is below the required 95%")
        print(f"\nAccuracy test passed. Model achieved {accuracy:.2f}% accuracy")

if __name__ == '__main__':
    unittest.main()
