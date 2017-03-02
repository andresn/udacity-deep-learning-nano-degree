import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork(object):

    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(-inputs))
    
    def sigmoid_prime(self, output):
        return output * (1 - output)
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights

        # (hidden nodes, input nodes)
        # (2, 56)
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        # (output nodes, hidden nodes)
        # (1, 2)
        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        
        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = self.sigmoid
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T # (56, 1)
        targets = np.array(targets_list, ndmin=2).T # (1, 1)
                
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) # (2, 56) * (56, 1) = (2, 1);
                                                                     # signals into hidden layer
        hidden_outputs = self.sigmoid(hidden_inputs) # (8, 1); signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) # (1, 2) * (2, 1) = (1,1);
                                                                             # signals into final output layer
        final_outputs = final_inputs # (1,1); f(x) = x; signals from final output layer
        
        #### Implement the backward pass here ####
        ### Backward pass ###
        
        # TODO: Output error        
        output_errors = targets - final_outputs # (1, 1) - (1, 1) = (1, 1); Output layer error is the difference
                                                # between desired target and actual output.
        # we don't multiply by gradient because f(x) = x; is 1. 
        
        # TODO: Backpropagated error
        hidden_errors = np.dot(output_errors, self.weights_hidden_to_output) # (1, 1) * (1, 2) = (1, 2) errors 
                                                                             # propagated to the hidden layer
        hidden_grad = self.sigmoid_prime(hidden_outputs) # (2, 1); hidden layer gradients
        
        hidden_error_term = hidden_errors.T * hidden_grad # (2, 1) * (2, 1) = ***(2, 1)***

        
        # TODO: Update the weights
        
        delta_weights_hidden_to_output = output_errors * hidden_outputs # (1, 1) * (2, 1) = ***(2, 1)***
        
        delta_weights_input_to_hidden = hidden_error_term.T * inputs # (1, 2) * (56, 1) = ***(56, 2)***;
                                                                     # Transposing the hidden_error_term because
                                                                     # this is more similar to exercise #16 - Implementing
                                                                     # Backpropagation

        self.weights_hidden_to_output += self.lr * delta_weights_hidden_to_output.T # (1, 2); update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_input_to_hidden.T # (2, 56); update input-to-hidden weights with gradient descent step

        
    def run(self, inputs_list):
        # Run a forward pass through the network

        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T # (56, 1)
                
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) # (2, 56) * (56, 1) = (2, 1);
                                                                     # signals into hidden layer
        hidden_outputs = self.sigmoid(hidden_inputs) # (8, 1); signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) # (1, 2) * (2, 1) = (1,1);
                                                                             # signals into final output layer
        final_outputs = final_inputs # (1,1); f(x) = x; signals from final output layer
        
        return final_outputs
