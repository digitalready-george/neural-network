import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

import time
from network import Network
tic = time.time()
net = Network([784, 40, 20, 20, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
toc = time.time()

elapsed_time = toc - tic
print(f"training time: {elapsed_time} seconds")