"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
"""

import numpy as np
from agents.base_agent import BaseAgent

class QLearningTrainedAgent(BaseAgent):
    """
    A Q-learning agent trained on the sailing environment.
    Uses a discretized state space and a lookup table for actions.
    """
    
    def __init__(self):
        """Initialize the agent with the trained Q-table."""
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # State discretization parameters
        self.position_bins = 8
        self.velocity_bins = 4
        self.wind_bins = 8
        
        # Q-table with learned values
        self.q_table = {}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
        self.q_table[(6, 4, 0)] = np.array([130.876 , 153.4555, 138.5399, 138.2441, 136.4966, 128.8462, 133.4192,
 137.3216, 138.0924])
        self.q_table[(5, 5, 0)] = np.array([ 3.988 ,  7.6689,  6.6485,  9.7704,  8.472 ,  4.0957,  8.0325, 25.6203,
  9.8326])
        self.q_table[(5, 5, 1)] = np.array([-4.8582,  0.    ,  0.    ,  0.    ,  5.7333,  0.    ,  0.    , -4.7761,
  1.1156])
        self.q_table[(6, 5, 0)] = np.array([130.5162, 151.703 , 136.303 , 133.5227, 132.0377, 133.4818, 128.796 ,
 128.7224, 133.5724])
        self.q_table[(6, 5, 1)] = np.array([-9.307 , -3.6589,  0.    , 56.7911,  0.    ,  0.    ,  0.    , -4.9154,
  0.7282])
        self.q_table[(6, 3, 0)] = np.array([145.1436, 154.0927, 140.3323, 144.0483, 143.9338, 142.8623, 139.5078,
 135.0727, 142.6338])
        self.q_table[(5, 4, 0)] = np.array([19.8515, 56.9758,  8.0904,  4.5639,  8.2895,  8.0684, 10.1854,  9.481 ,
  8.5047])
        self.q_table[(5, 4, 1)] = np.array([-4.2995, -4.9254,  0.    ,  0.    ,  0.    ,  0.    , 19.7113,  0.    ,
  0.5182])
        self.q_table[(5, 6, 0)] = np.array([ 1.87  ,  2.8787,  1.8777,  1.3038,  1.8979,  3.2682,  3.1791, 11.7118,
  1.3201])
        self.q_table[(5, 6, 1)] = np.array([ 0.    ,  0.    ,  0.2929,  0.    ,  0.3518,  0.    ,  0.2217, -4.0811,
  2.9769])
        self.q_table[(5, 7, 0)] = np.array([4.9511, 1.334 , 2.0652, 2.2937, 2.3744, 2.2064, 2.2949, 1.6324, 2.9318])
        self.q_table[(5, 7, 1)] = np.array([ 0.    ,  0.    , -4.9778,  0.    ,  0.2493,  0.    ,  0.5888, -4.1753,
  0.2588])
        self.q_table[(5, 0, 0)] = np.array([ 2.3423,  1.5962, -0.8051,  1.5934,  1.787 ,  1.4217,  2.4027,  7.3934,
  1.6736])
        self.q_table[(5, 0, 1)] = np.array([ 0.2762,  0.    ,  0.    ,  0.    ,  0.    , -0.0302,  0.    ,  0.    ,
  0.    ])
        self.q_table[(5, 1, 0)] = np.array([2.6085, 2.4089, 1.3703, 1.6816, 2.3521, 1.5482, 7.0614, 2.0082, 1.5959])
        self.q_table[(5, 2, 0)] = np.array([2.611 , 1.2322, 1.977 , 3.4763, 2.6054, 3.2518, 2.7957, 3.196 , 7.2461])
        self.q_table[(5, 2, 1)] = np.array([-4.3773, -4.8071,  0.0725,  0.    ,  0.    ,  0.188 ,  0.0778,  0.    ,
  0.8021])
        self.q_table[(6, 2, 0)] = np.array([126.8506, 150.5965, 122.8078, 117.1672, 119.7723, 115.2542,  82.863 ,
  94.698 , 126.6315])
        self.q_table[(6, 2, 1)] = np.array([ 0.    ,  1.6897,  0.    , 15.629 ,  0.    ,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(5, 1, 1)] = np.array([0.0323, 0.    , 0.1105, 0.    , 0.    , 0.    , 0.    , 0.    , 3.201 ])
        self.q_table[(6, 4, 1)] = np.array([ 0.    , 11.5264,  0.    ,  0.    ,  0.9688,  0.    ,  0.223 ,  0.4348,
  0.    ])
        self.q_table[(6, 6, 0)] = np.array([135.9336, 135.5313, 159.5267, 138.2716, 138.5424, 137.1279, 137.3162,
 139.5976, 136.8703])
        self.q_table[(6, 6, 1)] = np.array([-4.8609e+00,  8.7974e+01,  0.0000e+00,  0.0000e+00,  4.4268e-02,
 -4.9561e+00, -4.9706e+00, -4.9270e+00,  0.0000e+00])
        self.q_table[(6, 7, 1)] = np.array([ 3.5814e-02,  0.0000e+00,  6.5778e-01,  0.0000e+00,  0.0000e+00,
  1.0994e-01, -4.6504e+00,  4.1070e-01,  5.7168e+01])
        self.q_table[(6, 7, 0)] = np.array([142.5069, 161.7303, 144.2938, 143.1574, 140.3064, 142.9953, 139.9115,
 138.1012, 142.257 ])
        self.q_table[(6, 0, 0)] = np.array([149.1287, 143.5231, 164.6025, 133.387 , 130.9594, 134.1772, 124.9518,
 125.4334, 117.2655])
        self.q_table[(6, 1, 0)] = np.array([ 61.3694, 149.2184,  65.4566,  73.7518, 101.6108,  77.452 ,  71.0365,
  52.9195,  88.8036])
        self.q_table[(7, 3, 0)] = np.array([159.6523, 127.1996, 145.2504,  99.0901, 147.279 , 148.1855, 128.653 ,
 134.3351, 136.5048])
        self.q_table[(7, 4, 0)] = np.array([137.7997, 128.4126, 146.403 , 150.5178, 143.1503, 144.5013, 147.0783,
 166.2146, 148.4875])
        self.q_table[(7, 5, 0)] = np.array([ 81.3911,  94.2063,  81.442 , 115.3528, 106.3064, 102.1957, 177.2834,
 116.1351,  90.6008])
        self.q_table[(7, 6, 0)] = np.array([109.8255, 109.5134, 117.4284, 122.0497, 122.0755,  89.7338, 140.5983,
 176.5401, 111.1845])
        self.q_table[(7, 7, 0)] = np.array([ 63.2605,  43.5362,  63.061 ,  45.7977,  81.6466, 163.7652,  64.8102,
  58.5606,  69.8521])
        self.q_table[(7, 0, 0)] = np.array([164.1563, 165.5777, 164.7511, 162.624 , 156.3258, 156.8345, 211.6647,
 170.403 , 159.7815])
        self.q_table[(7, 1, 0)] = np.array([221.6238, 162.1767, 168.0699, 165.7287, 162.1632, 166.3209, 162.1956,
 165.4385, 162.2037])
        self.q_table[(7, 2, 0)] = np.array([159.3374, 158.324 , 160.6626, 153.0351, 157.7156, 157.235 , 147.8967,
 193.1198, 157.1598])
        self.q_table[(0, 2, 0)] = np.array([113.6471, 137.5896, 114.7363, 148.6805, 150.5354, 194.5897, 158.9422,
 140.2341, 130.3356])
        self.q_table[(0, 3, 0)] = np.array([121.4559, 126.9769, 129.5435, 161.3204, 116.8391, 120.1222, 131.6913,
 134.9609, 108.2535])
        self.q_table[(0, 4, 0)] = np.array([169.3992, 129.4718, 149.1076, 143.9572, 143.9732, 140.267 , 150.1546,
 152.6113, 138.789 ])
        self.q_table[(0, 5, 0)] = np.array([  0.4584,  44.5723,  18.1008,  16.9069,  18.8041, 171.9793,  17.6315,
   7.6746,  19.72  ])
        self.q_table[(0, 6, 0)] = np.array([2.8000e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.2085e+01,
 0.0000e+00, 0.0000e+00, 0.0000e+00])
        self.q_table[(6, 0, 1)] = np.array([ 0.0664,  0.0717,  0.    ,  4.2831, -5.0353,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(0, 0, 0)] = np.array([156.8298, 155.0066, 168.2503, 148.5187, 147.5385, 158.9736, 153.102 ,
 238.6855, 167.1982])
        self.q_table[(0, 1, 0)] = np.array([171.7604, 172.5624, 174.1484, 169.0442, 181.2991, 181.9687, 262.1135,
 186.7611, 180.2159])
        self.q_table[(6, 3, 1)] = np.array([ 3.1208, -4.5776,  0.6219,  0.1658,  6.4225, 22.6685, -4.9566,  0.    ,
  0.    ])
        self.q_table[(5, 3, 0)] = np.array([ 1.1996,  1.475 ,  0.9354,  2.0861,  1.9288,  4.3781,  2.9786, 36.8942,
  1.869 ])
        self.q_table[(4, 7, 0)] = np.array([9.2297, 2.9191, 1.3612, 1.7669, 1.318 , 1.5968, 2.2805, 2.1115, 1.0548])
        self.q_table[(4, 0, 0)] = np.array([12.6762,  4.6596,  2.9862,  2.3082,  3.0055,  2.2266,  1.8522,  4.2636,
  3.1349])
        self.q_table[(4, 1, 0)] = np.array([10.0343, 32.6117,  9.6361,  6.5029,  7.5506,  7.4262,  8.7137,  9.9819,
  7.797 ])
        self.q_table[(4, 2, 0)] = np.array([13.4737, 12.799 , 37.2518, 10.6276,  9.7829,  4.886 , 11.3064, 11.0867,
  9.5363])
        self.q_table[(4, 3, 0)] = np.array([11.7772,  1.9489,  2.0381,  2.9635,  2.2299,  1.9524,  1.5163,  2.4479,
  0.8338])
        self.q_table[(4, 4, 0)] = np.array([ 2.3139,  6.3454, 24.3742,  8.0434,  5.54  ,  2.6153,  4.7233,  6.4028,
  4.5148])
        self.q_table[(4, 5, 0)] = np.array([ 8.6103,  9.6088, 19.8432,  3.6716,  6.1299,  1.9188,  6.4139,  1.919 ,
  2.1404])
        self.q_table[(4, 6, 0)] = np.array([ 4.0045, 27.0096,  7.5097,  2.3775,  7.6085,  7.592 ,  1.4952,  4.5555,
 10.8713])
        self.q_table[(6, 1, 1)] = np.array([ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    , 13.7642,  0.    ,
  0.    ])
        self.q_table[(0, 7, 0)] = np.array([1.0685e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.5084e+01,
 0.0000e+00, 2.5698e-03, 0.0000e+00])
        self.q_table[(5, 3, 1)] = np.array([-4.9283,  0.    ,  0.    ,  0.    ,  0.    , 11.0742,  0.    ,  0.    ,
  0.    ])

    def discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table lookup."""
        # Extract position, velocity and wind from observation
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        
        # Discretize position (assume 128x128 grid)
        grid_size = 128
        x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)
        
        # Discretize velocity direction
        v_magnitude = np.sqrt(vx**2 + vy**2)
        if v_magnitude < 0.1:  # If velocity is very small, consider it as a separate bin
            v_bin = 0
        else:
            v_direction = np.arctan2(vy, vx)  # Range: [-pi, pi]
            v_bin = int(((v_direction + np.pi) / (2 * np.pi) * (self.velocity_bins-1)) + 1) % self.velocity_bins
        
        # Discretize wind direction
        wind_direction = np.arctan2(wy, wx)  # Range: [-pi, pi]
        wind_bin = int(((wind_direction + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins
        
        # Return discrete state tuple
        return (x_bin, y_bin, wind_bin)
        
    def act(self, observation):
        """Choose the best action according to the learned Q-table."""
        # Discretize the state
        state = self.discretize_state(observation)
        
        # Use default actions if state not in Q-table
        if state not in self.q_table:
            return 0  # Default to North if state not seen during training
        
        # Return action with highest Q-value
        return np.argmax(self.q_table[state])
    
    def reset(self):
        """Reset the agent for a new episode."""
        pass  # Nothing to reset
        
    def seed(self, seed=None):
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)
