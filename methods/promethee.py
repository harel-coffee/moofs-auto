import numpy as np
from PROMETHEE_Final_Rank_Figure import plot

def uni_cal(solutions_col, criteria_min_max, preference_function):
    """ solutions_col is the action performances array,
    c is the criteria min (0) or max (1) optimization array,
    and f is the preference	function array for a specific criterion ('u'
	for usual)
    """
    uni = np.zeros((solutions_col.shape[0], solutions_col.shape[0]))
    for i in range(np.size(uni, 0)):
        for j in range(np.size(uni, 1)):
            if i == j:
                uni[i, j] = 0
            elif preference_function == 'u':  # Usual preference function
                diff = solutions_col[j] - solutions_col[i]
                if diff > 0:
                    uni[i, j] = 1
                else:
                    uni[i, j] = 0

    if criteria_min_max == 0:
        uni = uni
    elif criteria_min_max == 1:
        uni = uni.T
    # positive, negative and net flows
    pos_flows = sum(uni, 1) / (uni.shape[0] - 1)
    neg_flows = sum(uni, 0) / (uni.shape[0] - 1)
    net_flows = pos_flows - neg_flows
    return net_flows


def promethee(solutions, criteria_min_max, preference_function, criteria_weights):
    """ solutions is the action performances array,
    c is the criteria min (0) or max (1)
	optimization array, d is the preference
	function array ('u' for usual),
    and w is the weights array
    """
    weighted_uni_net_flows = []
    total_net_flows = []
    for i in range(solutions.shape[1]):
        print(solutions[:, i:i + 1])
        weighted_uni_net_flows.append(criteria_weights[i] *
            uni_cal(solutions[:, i:i + 1], criteria_min_max[i], preference_function[i]))
    # net flows
    for i in range(np.size(weighted_uni_net_flows, 1)):
        k = 0
        for j in range(np.size(weighted_uni_net_flows, 0)):
            k = k + round(weighted_uni_net_flows[j][i], 5)
        total_net_flows.append(k)
    return np.around(total_net_flows, decimals = 4)

# main function
def main():
    # action performances array
    solutions = np.array([[-0.99, 0.5], [-0.9, 0.8], [-0.95, 0.6]])

    print(solutions.shape)
    solutions = [np.array([-0.69105691,  0.76262626]), np.array([-0.69105691,  0.76262626])]
    solutions = np.array(solutions)
    print(solutions)
    print(solutions.shape)

    # Normalization - chyba nie trzeba
    # x = np.zeros((3,2))
    # x[:,0] = [(float(i)-min(x_oryg[:,0]))/(max(x_oryg[:,0])-min(x_oryg[:,0])) for i in x_oryg[:,0]]
    # x[:,1] = [(float(i)-min(x_oryg[:,1]))/(max(x_oryg[:,1])-min(x_oryg[:,1])) for i in x_oryg[:,1]]
    # print(x)

    # criteria min (0) or max (1) optimization array
    criteria_min_max = ([0, 0])

    # preference function array
    preference_function = (['u', 'u'])

    # weights of criteria
    criteria_weights = np.array([0.5, 0.5])

    # final results
    # final_net_flows = promethee(solutions, criteria_min_max, preference_function, criteria_weights)
    # print("Global preference flows = ", final_net_flows)

if __name__ == '__main__':
    main()
