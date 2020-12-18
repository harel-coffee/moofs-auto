import numpy as np
from PROMETHEE_Final_Rank_Figure import plot

def uni_cal(x, c, f):
    """ x is the action performances array,
    c is the criteria min (0) or max (1) optimization array,
    and f is the preference	function array for a specific criterion ('u'
	for usual)
    """
    uni = np.zeros((x.shape[0], x.shape[0]))
    # print(uni)
    for i in range(np.size(uni, 0)):
        for j in range(np.size(uni, 1)):
            if i == j:
                uni[i, j] = 0
            elif f == 'u':  # Usual preference function
                diff = x[j] - x[i]
                # print(x[j], x[i], diff)
                if diff > 0:
                    # print(x[j], x[i])
                    # uni[i, j] = 1
                    uni[i, j] = diff
                else:
                    # print(x[j], x[i])
                    uni[i, j] = 0
    # print(uni)
    if c == 0:
        uni = uni
    elif c == 1:
        uni = uni.T
    # positive, negative and net flows
    pos_flows = sum(uni, 1) / (uni.shape[0] - 1)
    neg_flows = sum(uni, 0) / (uni.shape[0] - 1)
    net_flows = pos_flows - neg_flows
    print(net_flows)
    return net_flows


def promethee(x, c, d, w):
    """ x is the action performances array,
    c is the criteria min (0) or max (1)
	optimization array, d is the preference
	function array ('u' for usual),
    and w is the weights array
    """
    weighted_uni_net_flows = []
    total_net_flows = []
    for i in range(x.shape[1]):
        # print(x[:, i:i + 1])
        weighted_uni_net_flows.append(w[i] *
            uni_cal(x[:, i:i + 1], c[i], d[i]))
    # print(weighted_uni_net_flows)
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
    # x_oryg = np.array([[-0.99, 0.5], [-0.9, 0.8], [-0.95, 0.6], [-0.98, 0.6], [-0.95, 0.9], [-0.91, 0.3], [-0.95, 0.1]])
    x_oryg = np.array([[-0.99, 0.5], [-0.9, 0.8], [-0.95, 0.6]])

    # Normalization
    x = np.zeros((3,2))
    x[:,0] = [(float(i)-min(x_oryg[:,0]))/(max(x_oryg[:,0])-min(x_oryg[:,0])) for i in x_oryg[:,0]]
    x[:,1] = [(float(i)-min(x_oryg[:,1]))/(max(x_oryg[:,1])-min(x_oryg[:,1])) for i in x_oryg[:,1]]

    # criteria min (0) or max (1) optimization array
    c = ([0, 0])

    # preference function array
    d = (['u', 'u'])

    # weights of criteria
    w = np.array([0.5, 0.5])

    # final results
    final_net_flows = promethee(x, c, d, w)
    print("Global preference flows = ", final_net_flows)

if __name__ == '__main__':
    main()
