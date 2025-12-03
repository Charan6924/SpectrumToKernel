import numpy as np

def get_knots(control_points,degree = 3, x_min = 0.0, x_max = 0.5):
    inner = control_points - degree - 1
    start = 1e-3
    inner_knots = np.geomspace(start,x_max,inner + 2)[1:-1]
    pad_start = np.array([x_min]*(degree+1))
    pad_end = np.array([x_max]*(degree+1))

    return np.concatenate((pad_start,inner_knots,pad_end))

x = get_knots(20)
print(x)


'''[0.         0.         0.         0.         0.00167847 0.00281727
 0.00472871 0.00793701 0.01332204 0.02236068 0.03753178 0.06299605
 0.10573713 0.17747683 0.29788994 0.5        0.5        0.5
 0.5       ]'''