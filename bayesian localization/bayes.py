import numpy as np
import scipy.io

NUM_STEPS = 9

#Adapt this so robot doesn't go out of map

position_hist = []

position = np.array([
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

position_hist.append(position)

state_model = np.array([
    [0.10, 0.00, 0.00],
    [0.70, 0.10, 0.05],
    [0.15, 0.80, 0.15],
    [0.05, 0.10, 0.70],
    [0.00, 0.00, 0.10]
]);

measurement_model = np.array([
    [0.65, 0.10, 0.20, 0.05],
    [0.05, 0.45, 0.10, 0.20],
    [0.20, 0.10, 0.50, 0.15],
    [0.05, 0.30, 0.15, 0.55],
    [0.05, 0.05, 0.05, 0.05]
]);

input_seq = np.array([
    1, 1, 1, 0, 1, 1, 1, 1, 1
]);

observation_seq = ([
    'Null', 
    'Hallway', 
    'Wall', 
    'Wall', 
    'Closed Door', 
    'Wall', 
    'Wall', 
    'Wall', 
    'Open Door', 
    'Wall', 
    'Closed Door'
])

actual_seq = ([
    'Wall',
    'Hallway',
    'Wall',
    'Closed Door',
    'Wall',
    'Open Door',
    'Wall',
    'Closed Door',
    'Wall',
    'Hallway'
])
    

place_dict = {
    'Hallway': 0,
    'Closed Door': 1,
    'Wall': 2,
    'Open Door': 3,
    'Nothing': 4
}

#Give a state prediction - update
#p(x_k+1|z_0:k)

for i in range(9):
    new_position = np.zeros(14)
    for j in range(8):
        new_position[j + 2: j + 7] += \
        position[j]*state_model[:, input_seq[i] + 1]
        
    position = new_position[4:14]
    measurement = observation_seq[i + 1]
    
    #Update measurement
    for j in range(9):
        
        position[j] *= measurement_model[place_dict[measurement]]\
                                        [place_dict[actual_seq[j]]]
        
    #Normalize measurement
    position /= position.sum()
    
    position_hist.append(position)
    
    scipy.io.savemat('state_hist.mat', dict(position_hist = position_hist))
        
    
    

    



    




    
