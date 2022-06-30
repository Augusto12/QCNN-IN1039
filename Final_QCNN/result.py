# This generates the results of the bechmarking code

import Benchmarking


#########
# Here are possible combinations of benchmarking user could try.
# Unitaries = [U_TTN, U_5, U_6, U_9, U_13, U_14, U_15, U_SO4, U_SU4]
# U_num_params = [2, 10, 10, 2, 6, 6, 4, 6, 15]
# dataset = 'mnist' , 'fashion_mnist'
# circuit = 'QCNN', 'Hierarchical'
#########

Unitaries = [['U_TTN','U_TTN'],['U_TTN','U_5'],['U_TTN','U_6'],['U_TTN','U_9'],['U_TTN','U_13'],['U_TTN','U_14'],['U_TTN','U_15'],['U_TTN','U_SO4'],['U_TTN','U_SU4'],['U_TTN','U_SU4_no_pooling']]
U_num_params = [[2,2],[2,10],[2,10],[2,2],[2,6],[2,6],[2,4],[2,6],[2,15],[2,15]]
Encodings = ['resize256']
dataset = 'mnist'
classes = [0,1]
binary = False
cost_fn = 'cross_entropy'

for i in range(5):
    Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit='QCNN', cost_fn= cost_fn, binary=binary)


