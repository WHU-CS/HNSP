import numpy as np
import random
import argparse

def load_data(file):
    fr = open(file)
    link_list = []
    for line in fr.readlines():
        arr = line.strip().split()
        link_list.append([int(arr[0]), int(arr[1])])

    link_mat = np.array(link_list)
    max_node, min_node = link_mat.max(), link_mat.min()

    adj_matrix = np.zeros((max_node + 1, max_node + 1))
    for item in link_list:
        node_1 = item[0]
        node_2 = item[1]
        adj_matrix[node_1, node_2] = 1
        adj_matrix[node_2, node_1] = 1
    return adj_matrix, link_list


def add_noise(file_path, ratio):
    adj_matrix, link_list = load_data(file_path)
    node_size = len(adj_matrix)
    link_size = len(link_list)

    rest = link_size - 1
    for i in range(int(link_size * ratio)):
        index = random.randint(0, rest)
        del link_list[index]
        rest -= 1
    # add links
    count = int(link_size * ratio)
    while (True):
        node_1 = random.randint(0, node_size) - 1
        node_2 = random.randint(0, node_size) - 1
        if node_1 != node_2 and adj_matrix[node_1, node_2] == 0:
            count -= 1
            link_list.append([node_1, node_2])
        if count == 0:
            break
    output = file_path + '_' + str(ratio) + '.txt'

    out_fr = open(output, 'w')
    for item in link_list:
        node_1 = item[0]
        node_2 = item[1]
        out_fr.write(str(node_1) + ' ' + str(node_2) + '\n')

    out_fr.close()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='../dataset/email-Eu-netwok',
                    help='dataset')
parser.add_argument('--ratio', type=float, default=0.1,
                    help='noise ratio')

args = parser.parse_args()

add_noise(args.dataset, args.ratio)
