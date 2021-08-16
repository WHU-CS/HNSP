import numpy as np
import random
import math
import matplotlib.pylab as plt
from tqdm import tqdm
import argparse


# QPSO算法

class QPSO(object):
    def __init__(self, particle_num, particle_dim, alpha, iter_num, max_value, min_value, cn, pa, aa, katz,  links):
        '''
        定义类参数
        :param particle_num: 粒子群大小
        :param particle_dim: 粒子维度，对应代优化参数的个数
        :param alpha: 控制系数
        :param iter_num: 最大迭代次数
        :param max_value: 参数的最大值
        :param min_value: 参数的最小值
        '''

        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.iter_num = iter_num
        self.alpha = alpha
        self.max_value = max_value
        self.min_value = min_value
        self.adj = adj
        self.links = links
        self.cn = cn
        self.pa = pa
        self.aa = aa
        self.katz = katz

    # 粒子群初始化
    def swarm_origin(self):
        '''
        初始化粒子群中的粒子位置
        input:self(object):QPSO类
        output:particle_loc(list):粒子群位置列表
        '''
        particle_loc = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                a = random.random()
                tmp1.append(a)
            particle_loc.append(tmp1)
        return particle_loc

    # 计算适应度函数数值列表
    def fitness(self, particle_loc):
        '''
        计算适应度函数值
        :param particle_loc: 粒子群位置列表
        :return: fitness_value(list):适应度函数值列表
        '''
        fitness_value = []
        # 此处需要自行定义适应度的计算函数
        # 此处为例子
        for i in range(self.particle_num):
            # rbf_svm = svm.SVC(kernel='rbf', C = particle_loc[i][0], gamma = particle_loc[i][1])
            # cv_scores = cross_validation.cross_val_score(rbf_svm)
            auc = self.auc(index_1=particle_loc[i][0], index_2=particle_loc[i][1], index_3=particle_loc[i][2],
                           index_4=particle_loc[i][3], cn=self.cn, pa=self.pa, aa=self.aa, katz=self.katz,
                           links=self.links)
            fitness_value.append(auc)
        # 当前粒子群最优适应度函数值和对应的参数
        current_fitness = 0.0
        current_parameter = []
        for i in range(self.particle_num):
            if current_fitness < fitness_value[i]:
                current_fitness = fitness_value[i]
                current_parameter = particle_loc[i]
        return fitness_value, current_fitness, current_parameter

    # 粒子位置更新
    def update(self, particle_loc, gbest_parameter, pbest_parameter):
        '''
        粒子位置更新
        :param particle_loc: 粒子群位置列表
        :param gbest_parameter: 全局最优参数
        :param pbest_parameter: 每个粒子的历史最优值
        :return:particle_loc(list):新的粒子群位置列表
        '''
        Pbest_list = pbest_parameter
        # 计算mbest
        mbest = []
        total = []
        for l in range(self.particle_dim):
            total.append(0.0)
        total = np.array(total)

        for i in range(self.particle_num):
            total += np.array(Pbest_list[i])
        for j in range(self.particle_dim):
            mbest.append(list(total)[j] / self.particle_num)

        # 位置更新
        # Pbest_list更新
        for i in range(self.particle_num):
            a = random.uniform(0, 1)
            Pbest_list[i] = list(
                np.array([x * a for x in Pbest_list[i]]) + np.array([y * (1 - a) for y in gbest_parameter]))
        # 更新particle_loc
        for j in range(self.particle_num):
            mbest_x = []  # 存储mbest与粒子位置差的绝对值
            for m in range(self.particle_dim):
                mbest_x.append(abs(mbest[m] - particle_loc[j][m]))
            u = random.uniform(0, 1)
            if random.random() > 0.5:
                particle_loc[j] = list(
                    np.array(Pbest_list[j]) + np.array([self.alpha * math.log(1 / u) * x for x in mbest_x]))
            else:
                particle_loc[j] = list(
                    np.array(Pbest_list[j]) - np.array([self.alpha * math.log(1 / u) * x for x in mbest_x]))

        # 将更新后的量子位置参数固定在[MIN_VALUE, MAX_VALUE]内
        # 每个参数的位置列表
        parameter_list = []
        for i in range(self.particle_dim):
            tmp1 = []
            for j in range(self.particle_num):
                tmp1.append(particle_loc[j][i])
            parameter_list.append(tmp1)
        # 每个参数取值的最大值、最小值、平均值
        value = []
        for i in range(self.particle_dim):
            tmp2 = [max(parameter_list[i]), min(parameter_list[i])]
            value.append(tmp2)
        for i in range(self.particle_num):
            for j in range(self.particle_dim):
                particle_loc[i][j] = (particle_loc[i][j] - value[j][1]) / (value[j][0] - value[j][1]) * (
                            self.max_value - self.min_value) + self.min_value

        return particle_loc

    ## 2.4 画出适应度函数值变化图
    def plot(self, results):
        '''画图
        '''
        X = []
        Y = []
        for i in range(self.iter_num):
            X.append(i + 1)
            Y.append(results[i])
        plt.plot(X, Y)
        plt.xlabel('Number of iteration', size=15)
        plt.ylabel('Value of AUC', size=15)
        plt.title('QPSO_AUC parameter optimization')
        plt.show()

    # 主函数
    def main(self):
        results = []
        best_fitness = 0.0
        # 1.粒子群初始化
        particle_loc = self.swarm_origin()

        # 2 初始化gbest_parameter, pbest_parameter, fitness_value列表
        # 2.1 gbest_parameter
        gbest_parameter = []
        for i in range(self.particle_dim):
            gbest_parameter.append(0.0)
        # 2.2 pbest_parameter
        pbest_parameter = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                tmp1.append(0.0)
            pbest_parameter.append(tmp1)
        # 2.3 fitness_value
        fitness_value = []
        for i in range(self.particle_num):
            fitness_value.append(0.0)
        # 3 迭代
        for i in tqdm(range(self.iter_num)):
            # 3.1计算当前适应度函数值列表
            current_fitness_value, current_best_fitness, current_best_parameter = self.fitness(particle_loc)
            # 3.2 求当前的gbest_parameter、pbest_parameters和best_fitness
            for j in range(self.particle_num):
                if current_fitness_value[j] > fitness_value[j]:
                    pbest_parameter[j] = particle_loc[j]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                gbest_parameter = current_best_parameter

            results.append(best_fitness)

            # 3.3 更新fitnetss_value
            fitness_value = current_fitness_value
            # 3.4 更新粒子群
            particle_loc = self.update(particle_loc, gbest_parameter, pbest_parameter)

        # 4 结果展示
        results.sort()
        self.plot(results)
        print('final parameters are: ', gbest_parameter)
        #         file = open('parameter.txt','w')
        #         for item in gbest_parameter:
        #             file.write(item + ' ')
        return gbest_parameter

    def auc(self, index_1, index_2, index_3, index_4, cn, pa, aa, katz, links):

        similarity_matrix = index_1 * cn + index_2 * pa + index_3 * aa + index_4 * katz

        def cal_auc(sim):
            auc = 0
            #             linkset = random.sample(links,20000)
            for link in links:
                node_1 = link[0]
                node_2 = link[1]
                score_1 = sim[node_1, node_2]

                flag = True
                node_3, node_4 = 0, 0
                while (flag):
                    node_3 = np.random.randint(0, len(adj))
                    node_4 = np.random.randint(0, len(adj))
                    if node_3 != node_4 and adj[node_3, node_4] == 0:
                        flag = False
                score_2 = sim[node_3, node_4]

                if score_1 > score_2:
                    auc += 1
                if score_1 == score_2:
                    auc += 0.5
            auc = auc / len(links)
            return auc

        auc = cal_auc(similarity_matrix)
        return auc


def CN(adj):
    sim_matrix = np.dot(adj, adj)
    max_si = sim_matrix.max()
    min_si = sim_matrix.min()
    return (sim_matrix - min_si) / (max_si - min_si)


def PA(adj):
    deg_row = sum(adj)
    deg_row.shape = (deg_row.shape[0],1)
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


def JA(MatrixAdjacency_Train):
    JA_Train = sum(MatrixAdjacency_Train)
    JA_Train.shape = (JA_Train.shape[0], 1)
    MatrixAdjacency_Train_Log = MatrixAdjacency_Train / JA_Train
    MatrixAdjacency_Train_Log = np.nan_to_num(MatrixAdjacency_Train_Log)

    Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train_Log)

    return Matrix_similarity


def GLHN(MatrixAdjacency_Train):
    similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train)

    deg_row = sum(MatrixAdjacency_Train)
    deg_row.shape = (deg_row.shape[0], 1)
    deg_row_T = deg_row.T
    tempdeg = np.dot(deg_row, deg_row_T)
    temp = np.sqrt(tempdeg)

    np.seterr(divide='ignore', invalid='ignore')
    Matrix_similarity = np.nan_to_num(similarity / temp)

    return Matrix_similarity


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


if __name__ == '__main__':
    node_size = 10312
    file_path = 'blogCatalog3.txt'
    adj, links = graph2uadj(node_size, file_path)
    cn_s = CN(adj)
    aa_s = AA(adj)
    pa_s = PA(adj)
    katz_s = Katz(adj)
    ja_s = JA(adj)
    glhn_s = GLHN(adj)

    particle_num = 100
    particle_dim = 6
    iter_num = 20
    alpha = 0.6
    max_value = 1
    min_value = 0.00001
    qpso = QPSO(particle_num, particle_dim, alpha, iter_num, max_value, min_value, cn_s, pa_s, aa_s, katz_s, links)
    qpso.main()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--node_size', type=int)
args = parser.parse_args()

node_size = args.node_size
file_path = args.dataset
adj, links = graph2uadj(node_size, file_path)
cn_s = CN(adj)
aa_s = AA(adj)
pa_s = PA(adj)
katz_s = Katz(adj)
ja_s = JA(adj)
glhn_s = GLHN(adj)

particle_num = 100
particle_dim = 6
iter_num = 20
alpha = 0.6
max_value = 1
min_value = 0.00001
qpso = QPSO(particle_num, particle_dim, alpha, iter_num, max_value, min_value, cn_s, pa_s, aa_s, katz_s, links)
qpso.main()
