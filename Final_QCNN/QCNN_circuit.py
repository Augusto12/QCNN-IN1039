# Implementation of Quantum Convolutional Neural Network (QCNN) circuit structure.

import pennylane as qml
import unitary
import embedding

# Convolutional layers
def conv_layer1(U, params):
    U[0](params[0], wires=[0, 7])
    for i in range(0, 8, 2):
        U[1](params[1], wires=[i, i + 1])
    for i in range(1, 7, 2):
        U[0](params[0], wires=[i, i + 1])
def conv_layer2(U, params):
    U[1](params[1], wires=[0, 6])
    U[0](params[0], wires=[0, 2])
    U[0](params[0], wires=[4, 6])
    U[1](params[1], wires=[2, 4])
def conv_layer3(U, params):
    U[0](params, wires=[0,4])

# Pooling layers
def pooling_layer1(V, params):
    for i in range(0, 8, 2):
        V(params, wires=[i + 1, i])
def pooling_layer2(V, params):
    V(params, wires=[2,0])
    V(params, wires=[6,4])
def pooling_layer3(V, params):
    V(params, wires=[0,4])



def QCNN_structure(U, params, U_params):
    INDEX_SECOND_SUBLAYER = U_params[0] * 3 + 2 * 3

    ANSATZ = [unitary.Conv_Ansatze_Selector(U[0]),unitary.Conv_Ansatze_Selector(U[1])]

    param1 = [params[0:U_params[0]], params[INDEX_SECOND_SUBLAYER : INDEX_SECOND_SUBLAYER + U_params[1]]]
    param2 = [params[U_params[0]: 2 * U_params[0]], params[INDEX_SECOND_SUBLAYER + U_params[1]:]]
    param3 = params[2 * U_params[0]: 3 * U_params[0]]

    param4 = params[3 * U_params[0]: 3 * U_params[0] + 2]
    param5 = params[3 * U_params[0] + 2: 3 * U_params[0] + 4]
    param6 = params[3 * U_params[0] + 4: 3 * U_params[0] + 6]

    conv_layer1(ANSATZ, param1)
    pooling_layer1(unitary.Pooling_ansatz1, param4)

    conv_layer2(ANSATZ, param2)
    pooling_layer2(unitary.Pooling_ansatz1, param5)

    conv_layer3(ANSATZ, param3)
    pooling_layer3(unitary.Pooling_ansatz1, param6)


def QCNN_structure_without_pooling(U, params, U_params):
    INDEX_SECOND_SUBLAYER = U_params[0] * 3 + 2 * 3

    ANSATZ = [unitary.Conv_Ansatze_Selector(U[0]),unitary.Conv_Ansatze_Selector(U[1])]

    param1 = [params[0:U_params[0]], params[INDEX_SECOND_SUBLAYER : INDEX_SECOND_SUBLAYER + U_params[1]]]
    param2 = [params[U_params[0]: 2 * U_params[0]], params[INDEX_SECOND_SUBLAYER + U_params[1]:]]
    param3 = params[2 * U_params[0]: 3 * U_params[0]]

    conv_layer1(ANSATZ, param1)
    conv_layer2(ANSATZ, param2)
    conv_layer3(ANSATZ, param3)

def QCNN_1D_circuit(U, params, U_params):
    param1 = params[0: U_params]
    param2 = params[U_params: 2*U_params]
    param3 = params[2*U_params: 3*U_params]

    for i in range(0, 8, 2):
        U(param1, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(param1, wires=[i, i + 1])

    U(param2, wires=[2,3])
    U(param2, wires=[4,5])

    U(param3, wires=[3,4])



dev = qml.device('default.qubit', wires = 8)
@qml.qnode(dev)
def QCNN(X, params, U, U_params, embedding_type='Amplitude', cost_fn='cross_entropy'):

    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)

    if U[1] == 'U_SU4_no_pooling':
        QCNN_structure_without_pooling(U, params, U_params)
    
    QCNN_structure(U, params, U_params)

    if cost_fn == 'mse':
        result = qml.expval(qml.PauliZ(4))
    elif cost_fn == 'cross_entropy':
        result = qml.probs(wires=4)
    return result
