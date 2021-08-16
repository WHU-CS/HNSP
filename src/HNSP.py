import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
import argparse

def hsc(parameters, adj):
    def CN(adj):
        sim_matrix = np.dot(adj, adj)
        max_si = sim_matrix.max()
        min_si = sim_matrix.min()
        return (sim_matrix - min_si) / (max_si - min_si)

    def PA(adj):
        deg_row = sum(adj)
        deg_row.shape = (deg_row.shape[0], 1)
        deg_row_T = deg_row.T
        sim_matrix = np.dot(deg_row, deg_row_T)
        max_si = sim_matrix.max()
        min_si = sim_matrix.min()
        return (sim_matrix - min_si) / (max_si - min_si)

    def AA(MatrixAdjacency_Train):
        logTrain = np.log(sum(MatrixAdjacency_Train))
        logTrain = np.nan_to_num(logTrain)
        logTrain.shape = (logTrain.shape[0], 1)
        MatrixAdjacency_Train_Log = MatrixAdjacency_Train / logTrain
        MatrixAdjacency_Train_Log = np.nan_to_num(MatrixAdjacency_Train_Log)

        Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train_Log)

        max_si = Matrix_similarity.max()
        min_si = Matrix_similarity.min()

        return (Matrix_similarity - min_si) / (max_si - min_si)

    def Katz(MatrixAdjacency_Train):
        Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train)

        Parameter = 1
        Matrix_LP = np.dot(np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train), MatrixAdjacency_Train) * Parameter

        Matrix_similarity = np.dot(Matrix_similarity, Matrix_LP)
        max_si = Matrix_similarity.max()
        min_si = Matrix_similarity.min()

        return (Matrix_similarity - min_si) / (max_si - min_si)

    similarity_matrix = parameters[0] * CN(adj) + parameters[1] * PA(adj) + parameters[2] * AA(adj) + parameters[3] * Katz(adj)

    return similarity_matrix


def graph2uadj(node_size, file):
    adj = np.zeros((node_size,node_size))
    link_set = []
    fr = open(file)
    for line in fr.readlines():
        arr = line.strip().split()
        node_1 = int(arr[0])
        node_2 = int(arr[1])
        adj[node_1, node_2] = 1
        adj[node_2, node_1] = 1
        link_set.append([node_1, node_2])
    return adj, link_set


def NSP_emb(C, H, S, D, beta, step, d):
    M = np.random.random((len(C), d))
    U = np.random.random((len(C), d))

    losses = []

    for i in tqdm(range(step)):
        M = M * (np.dot((C * H * H), U) / np.dot((np.dot(M, U.T) * H * H), U))
        U = U * ((np.dot((C.T * H.T * H.T), M) + beta * np.dot(S, U)) /
                 (np.dot(np.dot(U, M.T) * H.T * H.T, M) + beta * np.dot(D, U)))
        loss = np.linalg.norm(((C - np.dot(M, U.T)) * H), ord=2, axis=None, keepdims=False) + beta * np.trace(
            np.dot(np.dot(U.T, (D - S)), U))
        losses.append(loss)
    return M, U, losses


def plot(results, size):
    X = []
    Y = []
    for i in range(size):
        X.append(i + 1)
        Y.append(results[i])
    plt.plot(X, Y)
    plt.xlabel('Number of iteration', size=15)
    plt.ylabel('loss', size=15)
    plt.title('NSP parameter optimization')
    plt.show()


def main(parameters, node_size, network_path, noise_network_path, alp, beta, epochs, d):
    adj, links = graph2uadj(node_size, network_path)
    print('construct similarity matrix...')
    similarity_matrix = hsc(parameters, adj)

    print('generate train set ...')

    X = []
    # NAN元素处理
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if np.isnan(similarity_matrix[i, j]):
                similarity_matrix[i, j] = 0
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if adj[i, j] != 0:
                X.append(similarity_matrix[i, j])

    print('train GMM ...')

    X = np.array(X).reshape(-1, 1)
    gmm = GaussianMixture(n_components=4).fit(X)

    adj_noise, links = graph2uadj(node_size, noise_network_path)

    c_adj = np.array(adj_noise)
    Us = gmm.means_
    Vars = gmm.covariances_

    print('construct correction matrix ...')

    for i in tqdm(range(len(adj))):
        for j in range(len(adj)):
            index = gmm.predict(similarity_matrix[i, j].reshape(1, 1))
            u = Us[index]
            var = Vars[index, 0, 0]
            if adj_noise[i, j] == 0 and similarity_matrix[i, j] - 100 * var > u:
                c_adj[i, j] = u
                c_adj[j, i] = u

            if adj_noise[i, j] == 1 and similarity_matrix[i, j] + 100 * var < u:
                c_adj[i, j] = alp * var / abs(similarity_matrix[i,j] - u)
                c_adj[j, i] = alp * var / abs(similarity_matrix[i,j] - u)

    h_matrix = np.ones(similarity_matrix.shape)
    d_matrix = np.zeros(similarity_matrix.shape)
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if adj[i, j] > 0:
                h_matrix[i, j] = 2
            if i == j:
                d_matrix[i, j] = np.sum(similarity_matrix[i, :])
                if d_matrix[i, j] == 0:
                    d_matrix[i, j] = 1

    print('embedding ...')
    M, U, loss = NSP_emb(c_adj, h_matrix, similarity_matrix, d_matrix, beta, epochs, d)

    plot(loss, 100)
    return U


def node_classification(label_file, embeddings, node_size):
    # 加载数据集
    fr = open(label_file)
    nodes = []
    labels = []
    for line in fr.readlines():
        arr = line.split(' ')
        node = int(arr[0])
        label = int(arr[1])
        nodes.append(node)
        labels.append(label)
        c = zip(nodes, labels)
        map = dict(c)

    Y = []
    for i in range(node_size):
        Y.append(map[i])
    X = embeddings
    Y = np.array(Y).reshape(-1, 1)

    unique, count = np.unique(Y, return_counts=True)
    data_count = dict(zip(unique, count))
    print(data_count)

    micros = []
    macros = []

    for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=1, stratify=Y)
        sc = StandardScaler()
        sc.fit(X_train)

        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        lr = LogisticRegression(C=100.0, random_state=1)
        lr.fit(X_train_std, y_train)

        y_pre = lr.predict(X_test_std)

        micro = precision_score(y_test, y_pre, average="micro")
        macro = precision_score(y_test, y_pre, average="macro")

        print('i = ' + str(i))
        print('micro = ', micro)
        print('macro = ', macro)

        micros.append(micro)
        macros.append(macro)
    return micros, macros


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--node_size', type=int)
parser.add_argument('--parameters', nargs='+', type=float)
parser.add_argument('--noise_network', type=str)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--d', type=int, default=100)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--alp', type=int, default=0.5)
parser.add_argument('--labels', type=str)

args = parser.parse_args()

embedding = main(args.parameters, args.node_size, args.dataset, args.noise_network, args.alp, args.beta, args.epochs, args.d)

micros, macros = node_classification(args.labels, embedding, args.node_size)
print('micros = ' + str(micros))
print('macros = ' + str(macros))