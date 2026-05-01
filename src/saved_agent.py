"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
"""

import numpy as np
#from evaluator.base_agent import BaseAgent
from src.agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    """
    A Q-learning agent trained on the sailing environment.
    Uses a discretized state space and a lookup table for actions.
    """
    
    def __init__(self):
        """Initialize the agent with the trained Q-table."""
        super().__init__()
        self.np_random = np.random.default_rng()
        self.grid_size = (128, 128)
        self.goal_position = np.array([self.grid_size[0] // 2, self.grid_size[1] - 1])
        self.learning_rate = 0.1
        self.discount_factor = 0.995
        self.exploration_rate =  0.5
        
        # State discretization parameters
        self.position_bins = 8
        self.goal_angle_bins = 8
        self.velocity_bins = 4    
        self.wind_angle_bins = 8
        self.danger_bins = 2
        
        # Q-table with learned values
        self.q_table = {}
        self._init_q_table()
        

    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
        self.q_table[(0, 4, 6, 4, 0)] = np.array([  9.8135,   8.3573,  11.3707,  38.3668,  41.7316,  16.441 , 135.7199,
  31.3498,  19.3321])
        self.q_table[(3, 4, 6, 4, 0)] = np.array([ 0.078 ,  0.8832,  2.963 ,  0.3498,  0.1184,  1.6551,  9.6078, 76.216 ,
  0.1092])
        self.q_table[(3, 5, 5, 5, 0)] = np.array([ 92.06  , 127.148 , 100.6068, 118.2307, 102.6235,  91.0265, 130.1534,
 111.3663, 117.4169])
        self.q_table[(3, 5, 5, 5, 1)] = np.array([-4.0828,  0.0921,  0.8667,  0.0463,  0.3631,  0.8659,  0.0338,  8.769 ,
  0.2236])
        self.q_table[(1, 5, 5, 5, 0)] = np.array([127.9253,  12.303 ,  13.3652,  16.1155,   9.1684,   3.7776,  26.1217,
   7.8666,   8.4448])
        self.q_table[(2, 5, 6, 5, 0)] = np.array([-8.2384,  3.2061,  0.155 ,  0.    ,  0.0433, -0.0144,  0.0534,  0.    ,
  0.1521])
        self.q_table[(3, 5, 6, 5, 0)] = np.array([ 1.9370e-01,  1.2728e-01,  0.0000e+00,  1.8545e-02, -6.3216e-03,
  1.8345e-01,  4.4748e+01,  0.0000e+00,  0.0000e+00])
        self.q_table[(3, 5, 6, 5, 1)] = np.array([-4.8579, -4.8054,  0.1709,  0.0394,  0.    ,  0.    , -4.9882,  0.    ,
  0.    ])
        self.q_table[(0, 3, 6, 3, 0)] = np.array([ 1.1798e+00,  1.4938e+00,  2.6756e-01, -1.6965e-02,  4.3717e+00,
  1.0254e+01,  9.5760e+00,  1.3792e+02,  1.8810e+01])
        self.q_table[(3, 4, 5, 4, 0)] = np.array([126.6512, 121.8077, 123.4374, 117.2517, 124.8255, 115.1411, 121.1666,
 136.2367, 124.3956])
        self.q_table[(0, 4, 5, 4, 0)] = np.array([136.2034,  48.3086,  67.3946,  57.5515,  56.8315,  42.8153,  60.4397,
  61.2994,  44.6113])
        self.q_table[(2, 4, 6, 4, 0)] = np.array([38.7107,  0.3672,  0.1138,  9.9788,  0.    ,  1.1514,  5.2096,  0.9331,
  7.5616])
        self.q_table[(1, 4, 5, 4, 0)] = np.array([ 1.0542e+02,  1.4394e+00, -2.0770e+00,  0.0000e+00,  4.7777e+00,
  0.0000e+00,  4.8598e-02,  9.6534e-01,  8.0214e+00])
        self.q_table[(0, 5, 5, 5, 0)] = np.array([113.6756, 108.6102, 115.045 , 114.7905, 106.0216,  98.2145, 110.4582,
 130.8473, 112.0419])
        self.q_table[(2, 5, 5, 5, 0)] = np.array([149.5522,  14.1983,  18.7498,  17.4463,  26.1105,  21.6026,  23.5591,
  18.364 ,  10.0099])
        self.q_table[(3, 6, 5, 6, 0)] = np.array([121.1045, 114.2768, 123.7972, 123.1981, 102.425 , 106.7025, 122.226 ,
 137.2452, 121.0365])
        self.q_table[(0, 6, 5, 6, 0)] = np.array([ 48.3433,  11.3409,  41.5737,  24.3122,  48.5429,  17.1148,  10.6128,
 134.8503,  22.5511])
        self.q_table[(3, 6, 5, 6, 1)] = np.array([-3.5585,  0.0935,  2.8579,  0.373 , -0.0058,  0.1164,  0.2455,  0.0647,
  0.6364])
        self.q_table[(1, 6, 5, 6, 0)] = np.array([ -2.0485, 126.2556,  10.3843,   6.1833,   0.1955,   0.9093,   5.6689,
   1.9944,   6.3619])
        self.q_table[(2, 6, 5, 6, 0)] = np.array([128.2406,   5.6228,  10.6718,   4.0328,   6.8214,   4.4537,   6.3465,
   6.8082,   3.996 ])
        self.q_table[(3, 7, 5, 7, 1)] = np.array([-4.9704,  0.6747,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(2, 4, 5, 4, 0)] = np.array([138.0338,  91.0282, 121.3231,  96.0725, 108.7357, 116.3397,  99.6113,
 124.2953,  97.7925])
        self.q_table[(3, 4, 5, 4, 1)] = np.array([ 0.    , -4.9293,  0.    ,  0.    ,  0.    ,  0.11  ,  0.1126, -4.8245,
  0.5184])
        self.q_table[(2, 4, 5, 4, 1)] = np.array([ 0.    ,  0.    , -4.9822,  0.    ,  0.    ,  1.3635,  0.    ,  0.    ,
  0.    ])
        self.q_table[(3, 3, 5, 3, 0)] = np.array([111.6616, 127.3494, 113.4415, 109.5163, 108.1957, 126.5272, 113.809 ,
 139.7223, 115.0365])
        self.q_table[(2, 3, 5, 3, 0)] = np.array([140.9701, 112.7787, 100.4636, 121.329 , 102.2678,  95.6591, 120.9985,
 109.7854, 109.2353])
        self.q_table[(2, 3, 5, 3, 1)] = np.array([2.2828, 0.3932, 0.    , 0.0527, 0.    , 0.    , 0.8259, 0.3597, 0.    ])
        self.q_table[(3, 3, 5, 3, 1)] = np.array([-7.7634, -3.5599, -4.2996, -4.4925,  6.8626,  1.7181,  0.    ,  1.2566,
  2.3923])
        self.q_table[(3, 7, 5, 7, 0)] = np.array([119.6215,  17.3203,  29.4721,  19.9023,  16.5377,  31.7708,  52.0397,
  45.2255,  34.1549])
        self.q_table[(0, 7, 5, 7, 0)] = np.array([ 28.2684,  20.4073,  13.5614,   4.2953,   6.6856,  21.9334,   3.7181,
 119.9191,   6.5792])
        self.q_table[(2, 7, 5, 7, 0)] = np.array([119.4367,   6.4946,  14.9814,   4.9049,  14.2879,  22.5073,  22.7694,
  35.8969,  13.8246])
        self.q_table[(3, 0, 5, 0, 0)] = np.array([  4.879 ,  28.7731,  21.7087,  29.3873,  16.3921,  28.2179,   3.8338,
 121.0932,  34.3278])
        self.q_table[(0, 0, 5, 0, 0)] = np.array([ 54.6652,  53.8631,  44.7921,  57.6903,  46.2674,  40.2332,  51.176 ,
 120.0439,  63.4988])
        self.q_table[(2, 0, 5, 0, 0)] = np.array([122.468 ,  19.6109,   7.0921,  37.7259,   3.1805,  10.5962,  26.6732,
  13.5287,  34.5776])
        self.q_table[(1, 0, 5, 0, 0)] = np.array([ 7.9372e-01,  3.4308e+00,  1.1193e+02, -4.7886e-03,  1.0871e+00,
  3.9412e-01,  6.8896e-01,  7.7916e+00,  1.5560e+00])
        self.q_table[(2, 1, 5, 1, 1)] = np.array([-4.7309,  0.    , -9.3613,  0.0839,  0.    ,  0.    ,  0.3679,  2.5312,
  0.    ])
        self.q_table[(3, 1, 5, 1, 1)] = np.array([-9.3908e+00, -4.9214e+00, -4.9819e+00, -7.3503e-03,  2.5653e-01,
  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00])
        self.q_table[(2, 1, 5, 1, 0)] = np.array([ 40.9848,  44.851 ,  42.4031,  25.2336,  35.2645,  37.8903,  37.0974,
 123.4914,  38.0834])
        self.q_table[(2, 3, 6, 3, 0)] = np.array([  0.    ,   0.2732,   0.    ,   0.    ,   1.7516,   0.8868,   0.956 ,
 210.1142,   0.    ])
        self.q_table[(1, 4, 6, 4, 0)] = np.array([0.1785, 0.058 , 0.2249, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(3, 4, 6, 4, 1)] = np.array([-4.8005,  0.    ,  0.    ,  1.1182,  0.    ,  0.0096,  0.    ,  0.    ,
  0.    ])
        self.q_table[(0, 5, 6, 5, 0)] = np.array([ 1.7783,  0.3156,  0.0777,  0.    , -0.0032,  0.    ,  0.    ,  0.    ,
  0.103 ])
        self.q_table[(1, 5, 6, 5, 0)] = np.array([ 0.    ,  0.    ,  0.    ,  0.    , -0.0229,  0.    ,  0.    ,  0.    ,
  0.1365])
        self.q_table[(3, 6, 6, 6, 1)] = np.array([-4.9227,  0.    ,  0.0685,  0.    ,  0.    ,  0.0074,  0.    ,  0.    ,
  0.    ])
        self.q_table[(0, 6, 6, 6, 0)] = np.array([-8.5061,  0.1868,  0.    ,  0.    , -0.0259, -0.0224,  0.    ,  0.057 ,
  0.0427])
        self.q_table[(1, 7, 5, 7, 0)] = np.array([ 3.5612,  0.8126,  1.2498,  1.9773,  0.1804,  0.4519,  3.8548,  1.6396,
 87.9547])
        self.q_table[(2, 2, 5, 2, 1)] = np.array([-4.8887, -3.9598, -4.8824,  0.3672,  0.    ,  0.    ,  5.7032, -4.9404,
  0.    ])
        self.q_table[(0, 2, 5, 2, 0)] = np.array([110.0189, 127.1141, 126.7003, 125.5608, 121.2639, 120.4883, 129.983 ,
 126.5582, 126.3465])
        self.q_table[(3, 2, 5, 2, 0)] = np.array([ 11.5562,   9.9218,  11.8927,   8.6038,  19.5106,  19.0214,  18.2407,
 132.9049,   9.5634])
        self.q_table[(0, 5, 6, 5, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(2, 5, 6, 5, 1)] = np.array([-4.9147,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(2, 5, 5, 5, 1)] = np.array([-4.6477, -4.9015,  0.    ,  0.    ,  5.1977,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(3, 2, 5, 2, 1)] = np.array([-4.9318, -4.9232,  0.    , -4.9039,  0.3276,  0.3794,  2.0945,  0.6872,
  0.    ])
        self.q_table[(0, 1, 5, 1, 0)] = np.array([120.7055, 116.6466,  97.9347, 114.8372, 113.9292, 113.0084, 117.1016,
 116.9995, 122.2227])
        self.q_table[(3, 6, 6, 6, 0)] = np.array([0.    , 0.0124, 0.    , 0.    , 0.    , 0.    , 0.0442, 0.2718, 0.    ])
        self.q_table[(1, 6, 6, 6, 0)] = np.array([0.0617, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(0, 7, 6, 7, 0)] = np.array([ 9.8506e+01,  5.6875e-02,  6.8945e-01, -1.2000e-01,  0.0000e+00,
  0.0000e+00, -3.5415e+00, -3.4962e+00,  1.3568e-02])
        self.q_table[(3, 7, 6, 7, 1)] = np.array([0.0311, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(2, 0, 5, 0, 1)] = np.array([ 0.1522,  0.    , -4.9795,  0.    ,  0.    ,  0.1956,  0.    ,  0.    ,
  0.8603])
        self.q_table[(2, 6, 5, 6, 1)] = np.array([0.23  , 0.    , 1.6957, 0.    , 0.2481, 0.    , 0.    , 0.5358, 0.    ])
        self.q_table[(1, 1, 5, 1, 0)] = np.array([124.4158,   1.9443,  24.6508,   7.2603,  17.1645,  17.8446,  10.8279,
  13.6347,   8.1694])
        self.q_table[(3, 1, 5, 1, 0)] = np.array([ 1.3354e+02, -4.2915e+00,  1.1633e+01,  5.7250e-03,  1.3771e-01,
  7.4760e+00,  0.0000e+00,  1.1681e-01,  2.5747e-01])
        self.q_table[(2, 2, 5, 2, 0)] = np.array([ 30.8831,  41.8778,  20.0289,  15.0737,  33.7934,  27.3683,  38.1461,
 131.1274,  40.219 ])
        self.q_table[(1, 2, 5, 2, 0)] = np.array([130.1526,  14.9027,  24.1357,  20.5726,   6.759 ,   1.9272,  10.6131,
  14.6338,  10.7211])
        self.q_table[(0, 3, 5, 3, 0)] = np.array([ 75.8077,  81.3832,  28.444 ,  67.3958,  69.6442,  56.0561,  75.9303,
 138.3739,  48.8931])
        self.q_table[(3, 5, 4, 5, 0)] = np.array([119.5285, 146.2582, 127.5296, 124.119 , 123.7381, 120.5798, 117.4625,
 139.0709, 121.4883])
        self.q_table[(2, 5, 4, 5, 0)] = np.array([ 91.0138, 117.7381, 120.3583, 110.4686, 108.8746,  93.947 , 107.3566,
 145.8897, 124.0107])
        self.q_table[(0, 5, 4, 5, 0)] = np.array([ 47.1972, 146.6568,  27.281 ,  28.0248,  47.7685,  21.6897,  51.456 ,
  46.5742,  38.1995])
        self.q_table[(3, 6, 4, 6, 0)] = np.array([113.5691, 144.937 , 119.5982, 129.5203, 103.6591, 110.6957, 125.1694,
 122.2144, 130.295 ])
        self.q_table[(0, 6, 4, 6, 0)] = np.array([145.3923,  21.2958,  22.814 ,   8.8482,  16.7882,   9.3071,  16.3911,
   1.0518,  16.4939])
        self.q_table[(3, 7, 4, 7, 0)] = np.array([ 48.3739,  40.7201,  63.2818, 125.6043,  30.8668,  30.3959,  40.0677,
  43.8325,  49.5867])
        self.q_table[(0, 7, 4, 7, 0)] = np.array([111.6992, 123.9453, 105.7379, 121.2061, 108.2491, 104.4681, 111.8858,
 110.6291, 112.955 ])
        self.q_table[(1, 7, 4, 7, 0)] = np.array([  6.9785,   4.2421,  10.3284,   3.5368,   6.2106,   8.4463,   9.4582,
  12.9267, 121.8983])
        self.q_table[(2, 7, 4, 7, 0)] = np.array([104.6605, 153.1337,  78.6374,  85.7589,  79.3554,  87.1991,  90.86  ,
  91.6927,  77.1373])
        self.q_table[(3, 0, 4, 0, 0)] = np.array([114.6843,  93.6126,  95.6929,  98.5371,  96.6318,  84.143 ,  96.8906,
  86.532 ,  81.6006])
        self.q_table[(1, 0, 4, 0, 0)] = np.array([113.2852,   4.0742,   6.5448,   6.5447,   8.0131,   7.1787,   0.2875,
   2.8266,   1.5139])
        self.q_table[(0, 0, 4, 0, 0)] = np.array([114.1362,  96.7921, 105.5091,  94.6022, 100.1326, 102.2301, 100.569 ,
 107.5138, 107.8966])
        self.q_table[(3, 1, 4, 1, 0)] = np.array([ 19.1813,   7.1537,   1.1161,   0.2604,   7.1763,  29.9565,  14.0189,
  27.7146, 120.7641])
        self.q_table[(0, 1, 4, 1, 0)] = np.array([ 71.9824,  62.4242,  74.265 , 116.1165,  81.5891,  79.3615,  79.0374,
  41.5237,  92.7416])
        self.q_table[(2, 1, 4, 1, 0)] = np.array([100.2251, 123.8947,  96.3459, 105.4822,  68.7954,  86.068 ,  90.7497,
  88.2078,  95.8614])
        self.q_table[(1, 1, 4, 1, 0)] = np.array([6.2700e+00, 4.8101e-01, 1.2079e+02, 5.6898e-01, 9.0633e-02, 2.7882e-01,
 1.4810e+01, 1.0631e+01, 5.3414e+00])
        self.q_table[(0, 2, 4, 2, 0)] = np.array([5.6867e-01, 2.6183e-01, 1.1911e+02, 2.8318e+00, 2.4365e+00, 4.8391e-02,
 2.4355e+00, 4.2040e+00, 3.1705e-01])
        self.q_table[(2, 2, 4, 2, 0)] = np.array([ 35.6564,  19.0872,  18.4382, 131.5248,  12.9227,  21.3368,  31.6876,
  30.4456,  23.7496])
        self.q_table[(1, 2, 4, 2, 0)] = np.array([2.9522e+00, 0.0000e+00, 1.0011e-01, 7.1508e+01, 5.2366e+00, 0.0000e+00,
 1.9949e-02, 5.8916e-01, 7.2445e+00])
        self.q_table[(3, 2, 4, 2, 0)] = np.array([2.2431e-02, 1.7321e-01, 2.2483e-01, 8.9866e+00, 3.0430e-01, 7.1469e-01,
 4.1767e+00, 9.2790e+01, 1.0256e-01])
        self.q_table[(3, 3, 4, 3, 0)] = np.array([120.8457, 103.3886, 105.8159, 141.2981,  87.7058,  90.6774,  90.3764,
 111.8617,  91.0843])
        self.q_table[(2, 3, 4, 3, 0)] = np.array([ 19.704 ,   7.4318,  53.6237, 138.2918,  17.0026,  45.8656,  29.4744,
  35.9387,  21.8334])
        self.q_table[(0, 3, 4, 3, 0)] = np.array([1.3225e+02, 9.0105e+00, 1.6664e+01, 5.0436e-02, 8.0805e+00, 0.0000e+00,
 8.2687e+00, 6.8255e+00, 3.6373e+00])
        self.q_table[(3, 4, 4, 4, 0)] = np.array([115.6816, 118.6181, 116.6854, 141.3873, 112.3357, 122.1404, 108.2191,
 114.5039, 103.2623])
        self.q_table[(2, 4, 4, 4, 0)] = np.array([107.9316, 108.2448, 138.0054,  91.3316,  90.0123, 106.2693, 102.7748,
 102.6361,  71.3247])
        self.q_table[(0, 4, 4, 4, 0)] = np.array([ 48.9637, 142.1794,  60.9214,  63.2494,  31.5548,  30.6668,  48.5108,
  27.1132,  78.2565])
        self.q_table[(2, 6, 4, 6, 0)] = np.array([142.1403, 112.613 , 116.1552, 109.8334, 105.556 ,  92.6219, 114.31  ,
 104.44  , 105.6307])
        self.q_table[(2, 0, 4, 0, 0)] = np.array([116.4142,  38.8486,  33.366 ,  45.5822,   1.6335,  28.9468,  15.5802,
  44.9322,  35.2155])
        self.q_table[(2, 1, 6, 1, 0)] = np.array([-3.7083,  0.    ,  0.    ,  0.    , -0.019 ,  0.    ,  0.0147,  0.    ,
 12.8993])
        self.q_table[(3, 1, 6, 1, 0)] = np.array([ 8.5977e+01,  0.0000e+00, -1.6401e-02, -4.0462e-02, -1.9300e-02,
  1.1528e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00])
        self.q_table[(1, 1, 6, 1, 0)] = np.array([-4.9695,  0.0222,  0.    , -0.0206,  0.    ,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(0, 1, 6, 1, 0)] = np.array([43.4275,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
  0.3994])
        self.q_table[(3, 1, 7, 1, 0)] = np.array([5.9228e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.2426e-04,
 0.0000e+00, 0.0000e+00, 0.0000e+00])
        self.q_table[(3, 0, 0, 0, 0)] = np.array([ 1.2890e-02, -3.5950e-02, -3.7097e-02, -7.7166e-02, -2.0498e-03,
  9.8426e-04,  1.3774e+01,  0.0000e+00,  0.0000e+00])
        self.q_table[(2, 0, 0, 0, 0)] = np.array([ 0.0016,  0.    , -0.0192,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(2, 1, 7, 1, 0)] = np.array([2.4703, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(3, 1, 0, 1, 0)] = np.array([ 1.4021e+00,  2.6586e-02,  0.0000e+00, -4.0552e-02,  3.0458e+01,
  9.9130e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00])
        self.q_table[(0, 1, 0, 1, 0)] = np.array([ 2.9465e-01, -3.7330e-02,  1.8490e-01,  6.7724e-02,  0.0000e+00,
  3.0215e+00,  9.4863e+01,  1.4610e+00,  2.0127e+00])
        self.q_table[(2, 2, 0, 2, 0)] = np.array([0.0003, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(3, 2, 0, 2, 0)] = np.array([2.2099e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
 2.0200e-02, 1.0022e+01, 0.0000e+00])
        self.q_table[(0, 2, 0, 2, 0)] = np.array([0.0022, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(1, 3, 5, 3, 0)] = np.array([3.0356e+00, 1.2874e+02, 7.6628e+00, 8.7568e-01, 2.8157e+00, 9.0955e+00,
 3.2652e+00, 3.7108e-01, 1.5579e-02])
        self.q_table[(1, 6, 4, 6, 0)] = np.array([1.0200e+01, 0.0000e+00, 9.8061e-03, 6.8224e+00, 9.2371e+00, 1.9267e-01,
 8.2877e+00, 0.0000e+00, 1.2904e+02])
        self.q_table[(1, 3, 4, 3, 0)] = np.array([55.1534,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.1666,
  0.    ])
        self.q_table[(2, 2, 6, 2, 0)] = np.array([1.4661e+02, 3.6549e-02, 4.1909e-01, 4.3498e-01, 7.3909e-02, 1.3280e-01,
 0.0000e+00, 0.0000e+00, 6.1048e-02])
        self.q_table[(3, 3, 6, 3, 0)] = np.array([10.2752,  3.5906,  1.3593,  0.218 ,  0.165 ,  0.    ,  0.7738,  1.7417,
  0.    ])
        self.q_table[(3, 2, 6, 2, 0)] = np.array([6.1994e+01, 2.5852e+00, 0.0000e+00, 4.8507e-01, 7.5753e-03, 0.0000e+00,
 3.3984e-03, 3.1097e+00, 4.2705e-02])
        self.q_table[(0, 2, 6, 2, 0)] = np.array([ 0.063 ,  0.1364,  2.5716,  0.    , -0.0186,  0.    ,  0.    ,  0.    ,
  0.4173])
        self.q_table[(0, 6, 6, 6, 1)] = np.array([0.0131, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(2, 6, 6, 6, 0)] = np.array([0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0305])
        self.q_table[(3, 7, 6, 7, 0)] = np.array([0.0324, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(2, 7, 6, 7, 0)] = np.array([ 7.4621e+01,  0.0000e+00,  0.0000e+00, -4.5512e-02, -4.8406e-02,
  0.0000e+00,  0.0000e+00,  7.7837e-03,  0.0000e+00])
        self.q_table[(2, 0, 6, 0, 0)] = np.array([ 4.2087e+01,  3.5990e-02,  0.0000e+00,  0.0000e+00,  0.0000e+00,
 -2.7244e-02,  0.0000e+00,  0.0000e+00,  0.0000e+00])
        self.q_table[(0, 0, 6, 0, 0)] = np.array([7.5476e-02, 5.5570e-02, 2.4246e-01, 2.5661e+00, 2.9190e+00, 7.6595e+00,
 1.1664e+02, 0.0000e+00, 0.0000e+00])
        self.q_table[(0, 0, 6, 0, 1)] = np.array([0.0684, 0.    , 0.    , 0.    , 0.    , 0.    , 0.0145, 0.    , 0.    ])
        self.q_table[(3, 1, 6, 1, 1)] = np.array([-4.9316,  0.    ,  0.    ,  0.    ,  0.    , -0.0074,  0.    ,  0.    ,
  0.    ])
        self.q_table[(2, 4, 6, 4, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 6, 6, 6, 1)] = np.array([-4.9484,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(1, 5, 4, 5, 0)] = np.array([ 2.097 ,  4.5404,  0.    ,  0.    ,  0.    ,  0.    , 85.5817,  0.    ,
  0.2435])
        self.q_table[(0, 7, 6, 7, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 0, 6, 0, 0)] = np.array([ 0.    ,  0.    ,  0.    ,  0.    , -0.0169,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(1, 0, 6, 0, 0)] = np.array([0.0633, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(3, 0, 7, 0, 0)] = np.array([0.033 , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0003, 0.    ])
        self.q_table[(3, 7, 0, 7, 0)] = np.array([0.0014, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(1, 0, 7, 0, 0)] = np.array([0.0032, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(0, 0, 7, 0, 0)] = np.array([ 0.   , -0.023,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ])
        self.q_table[(2, 7, 0, 7, 0)] = np.array([0.0012, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(0, 0, 0, 0, 0)] = np.array([2.5262e+00, 0.0000e+00, 0.0000e+00, 2.2936e-02, 0.0000e+00, 7.6858e-04,
 0.0000e+00, 0.0000e+00, 0.0000e+00])
        self.q_table[(1, 7, 6, 7, 0)] = np.array([0.02, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ])
        self.q_table[(3, 2, 7, 2, 0)] = np.array([ 0.0668,  0.    ,  0.    , -0.027 ,  0.    ,  0.0009,  0.    ,  0.    ,
  0.    ])
        self.q_table[(0, 1, 7, 1, 0)] = np.array([0.0251, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(2, 0, 7, 0, 0)] = np.array([ 0.0032,  0.    , -0.0387,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(2, 1, 0, 1, 0)] = np.array([4.1798, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(3, 2, 6, 2, 1)] = np.array([-4.9292,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
  0.    ])
        self.q_table[(1, 2, 6, 2, 0)] = np.array([0.    , 0.9277, 0.    , 0.7804, 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(2, 2, 7, 2, 0)] = np.array([0.0149, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(0, 2, 7, 2, 0)] = np.array([0.0134, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(1, 2, 5, 2, 1)] = np.array([0.3119, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(3, 3, 6, 3, 1)] = np.array([1.0687, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(0, 3, 6, 3, 1)] = np.array([0.0357, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(0, 4, 6, 4, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(2, 7, 5, 7, 1)] = np.array([0.4424, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ])
        self.q_table[(1, 4, 4, 4, 0)] = np.array([0.    , 0.    , 0.    , 6.6105, 0.    , 0.    , 0.    , 0.    , 0.    ])

    
        
    def discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table lookup."""
        
        
        goal_x, goal_y = self.goal_position
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        grid_size = 128
        #x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        #y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)
        
        # angle to goal
        dx, dy = goal_x - x, goal_y - y
        angle_to_goal = np.arctan2(dy, dx)
        goal_bin = int(((angle_to_goal + np.pi) / (2 * np.pi) * self.goal_angle_bins)) % self.goal_angle_bins
        
        # relative wind
        wind_angle = np.arctan2(wy, wx)
        rel_wind = (wind_angle - angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        wind_bin = int(((rel_wind + np.pi) / (2 * np.pi) * self.wind_angle_bins)) % self.wind_angle_bins
        
        # discretize velocity direction (ignoring magnitude for simplicity)
        v_magnitude = np.sqrt(vx**2 + vy**2)
        if v_magnitude < 0.1:  # If velocity is very small, consider it as a separate bin
            v_bin = 0
        else:
            v_direction = np.arctan2(vy, vx)  # Range: [-pi, pi]
            v_bin = int(((v_direction + np.pi) / (2 * np.pi) * (self.velocity_bins-1)) + 1) % self.velocity_bins

        # danger radar
        danger_bin = 0
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 0.1:
            look_x = int(np.clip(x + vx * 2, 0, 127))
            look_y = int(np.clip(y + vy * 2, 0, 127))
            map_idx = (6 + 32768) + (look_y * 128 + look_x)
            if map_idx < len(observation) and observation[map_idx] == 1:
                danger_bin = 1
        
        
        return (v_bin, wind_bin, goal_bin, wind_bin, danger_bin)
        
    def act(self, observation):
        """Choose the best action according to the learned Q-table."""
        # Discretize the state
        state = self.discretize_state(observation)
        
        # Use default actions if state not in Q-table
        if state not in self.q_table:
            return np.random.integers(0,9)  
        
        # Return action with highest Q-value
        return np.argmax(self.q_table[state])
    
    def reset(self):
        """Reset the agent for a new episode."""
        pass  # Nothing to reset
        
    def seed(self, seed=None):
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)



#evaluation of performances
from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario
ql_agent = MyAgent()

np.random.seed(42)
ql_agent.seed(42)
ql_agent.exploration_rate = 0

for scenario in ['training_1', 'training_2', 'training_3']:

    # Create test environment
    test_env = SailingEnv(**get_wind_scenario(scenario))

    # added this because goal enters the agent in my implementation
    goal = test_env.goal_position.copy()
    print("GOAL", goal)

    # Test parameters
    num_test_episodes = 5
    max_steps = 500

    print("Testing the trained agent on 5 new episodes...")
    # Testing loop
    for episode in range(num_test_episodes):
        # Reset environment
        observation, info = test_env.reset(seed=1000 + episode)  # Different seeds from training

        
        total_reward = 0
        
        for step in range(max_steps):
            # Select action using learned policy
            action = ql_agent.act(observation)
            observation, reward, done, truncated, info = test_env.step(action)
            
            total_reward += reward
            
            # Break if episode is done
            if done or truncated:
                break
        
        print(f"Test Episode {episode+1}: Steps={step+1}, Reward={total_reward}, " +
            f"Position={info['position']}, Goal reached={done}")