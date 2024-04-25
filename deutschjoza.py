import numpy as np
from math import sqrt

# Python Implementation of deutch-joza algorithm, based on github example:
# https://github.com/MyEntangled/MyEntangled-Blog/blob/master/Quantum%20Programming%20Projects/Deutsch-Jozsa%20algorithm.ipynb

def getTensor(matrices):
    product = matrices[0]
    for matrix in matrices[1:]:
        product = np.kron(product,matrix)  ## np.kron stands for Kronecker product
    return product

def U(n, f_map):
    """Generate an oracle matrix based on the given function mapping."""
    num_qubits = n + 1
    U = np.zeros((2**num_qubits, 2**num_qubits)) # Start with a matrix of zeroes.
    
    # Quantum state looks like IN-IN-...-IN-ANCILLA
    for input_state in range(2**num_qubits): # For each possible input
        input_string = input_state >> 1 # remove ANCILLA
        output_qubit = (input_state & 1) ^ (f_map[input_string]) # remove IN, XOR with f(IN)
        output_state = (input_string << 1) + output_qubit # the full state, with new OUT
        U[input_state, output_state] = 1 # set that part of U to 1

    return U

def measure(n, state):
    measurement = np.zeros(2**n)  # Initialize measurement result for n qubits in the first register
    for index, value in enumerate(state):
        measurement[index >> 1] += value * value  ## As the ancilla qubit is discarded, probabilities of the same kind, ie 100 and 101 will be combined

    # Last step: Determine the type of function f
    # f is constant if the probability of measuring |0> is positive
    if (abs(measurement[0]) > 1e-10): 
        print("The function is constant.")
    else:
        print("The function is balanced.")
        
def Deutsch_Jozsa(n, f_map):
    num_qubits = n + 1  # Plus one qubit and the second register, can be called as ancilla qubit
    state_0 = np.array([[1],[0]])  # Standard state |0> as a column vector
    I_gate = np.array([[1,0], [0,1]])  # Identity gate
    X_gate = np.array([[0,1], [1,0]])  # NOT gate
    H_gate = np.array([[1,1], [1,-1]])/sqrt(2)  # Hadamard gate
    
    ancilla = np.dot(X_gate, state_0)  # Create state |1> assigned to the ancilla
    
    # Create the a Hadamard transformation for all qubits and the state |ψ_0> 
    listStates = []
    listGates_H = []
    for i in range(n):
        listStates.append(state_0)
        listGates_H.append(H_gate)
    listStates.append(ancilla)
    listGates_H.append(H_gate)
    psi_0 = getTensor(listStates)
    composite_H = getTensor(listGates_H)
    
    # |ψ_1> is the dot product of the Hadamard transformation and |ψ_0>  
    psi_1 = np.dot(composite_H, psi_0)

    # Apply the oracle to |ψ_1>
    psi_2 = np.dot(U(n, f_map), psi_1)

    # H on all again
    psi_3 = np.dot(composite_H, psi_2)

    measure(n, psi_3)
    
def main():
    n = [2,3,3]  # Input the number of qubits
    f_map = [[0,0,1,1],
             [1,1,1,1,1,1,1,1],
             [1,0,0,1,1,0,1,0],]  # Input the mapping functions
    for index, value in enumerate(n):
        Deutsch_Jozsa(n[index], f_map[index])  # Algorithm executed here

if __name__ == "__main__":
    main()