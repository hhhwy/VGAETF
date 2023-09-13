import pandas as pd
import numpy as np
import torch
import networkx as nx


def data_split():
    columns = ['drugname1', 'drugname2', 'disease']
    all_triplets = pd.read_csv('../data/drug_drug_disease.csv', header=None, sep=',')
    all_triplets.columns = columns
    idx = np.arange(len(all_triplets))
    np.random.shuffle(idx)
    test_idx = idx[:int(len(all_triplets)*0.1)]
    train_val_idx = idx[int(len(all_triplets)*0.1):]
    #
    test_data = all_triplets.loc[test_idx].reset_index(drop=True)
    train_val_data = all_triplets.loc[train_val_idx].reset_index(drop=True)
    test_data.to_csv('../data/ddd_test.csv', sep=',', header=None, index=None)
    train_val_data.to_csv('../data/ddd_train.csv', sep=',', header=None, index=None)


def load_data():
    # num of relation
    with open('../data/relations.dict') as f:
        relation2id = dict()
        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    relation_num = len(relation2id)
    # num of drugs
    with open('../data/entities.dict') as f:
        entity2id = dict()
        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
    entity_num = len(entity2id)
    # all triplets
    # data_split()
    columns2 = ['Head', 'Tail', 'Relation']
    test_pos_tri = pd.read_csv('../data/ddd_test.csv', names=columns2, header=None, index_col=None)
    test_pos_tri = test_pos_tri.values.tolist()
    train_val_tri = pd.read_csv('../data/ddd_train.csv', names=columns2, header=None, index_col=None)
    train_val_tri = train_val_tri.values.tolist()
    train_val_tri = np.array(train_val_tri, dtype=np.int64)
    test_pos_tri = np.array(test_pos_tri, dtype=np.int64)
    feature_dim = entity_num
    features = torch.nn.Parameter(torch.FloatTensor(feature_dim, feature_dim))
    torch.nn.init.uniform_(features, a=-1, b=1)

    return features, train_val_tri, test_pos_tri, entity_num, relation_num


def get_symmetry_triplets(triplets):
    triplets_s, triplets_o, triplets_r = triplets.T
    triplets_symmetry = np.stack([triplets_o, triplets_s, triplets_r], axis=1)
    triplets_ = np.vstack((triplets, triplets_symmetry))
    return triplets_


def get_triplets(edges, false_edges):
    triplets = torch.LongTensor(np.concatenate((edges, false_edges)))
    triplets = triplets.cuda()
    return triplets


def negative_sampling(pos_samples, num_entity, relation_num):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch
    neg_samples = np.tile(pos_samples, (1, 1))
    # num_entity: 在train三元组batch中的实体的id，choices是values的下标
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    # 把正样本中的s替换掉，把随机在正样本中选择一半的正样本把它的s的id替换成随机生成的实体id
    neg_samples[subj, 0] = values[subj]
    # 把正样本剩下的一半的o替换成随机生成的实体id
    neg_samples[obj, 1] = values[obj]
    return neg_samples


def generate_negative_data(triplets, num_nodes, relation_num):
    false_triplets = negative_sampling(triplets, num_nodes, relation_num)

    return false_triplets


def get_labels(pos_triplets, neg_triplets):
    # labels
    labels = np.zeros(len(pos_triplets) + len(neg_triplets))
    labels[:len(pos_triplets)] = 1
    labels = torch.from_numpy(labels).cuda()
    return labels


def get_test_tri_label(test_pos_triplets, entity_num, relation_num):
    test_pos_triplets = get_symmetry_triplets(test_pos_triplets)
    test_false_triplets = generate_negative_data(test_pos_triplets, entity_num, relation_num)
    test_triplets = get_triplets(test_pos_triplets, test_false_triplets)
    test_labels = get_labels(test_pos_triplets, test_false_triplets)
    return test_triplets, test_labels


def generate_all_heteroGraph(graph, drug_node_num, relation_num):
    G = graph.copy()
    d_num = len(G.nodes())
    columns = ['Source', 'Target']
    data_dd = pd.read_csv('../data/drug_drug.csv', names=columns, header=None)
    data_jj = pd.read_csv('../data/disease_disease.csv', names=columns, header=None)
    data_dd_list = data_dd.values.tolist()
    data_jj_list = data_jj.values.tolist()
    dis_node_list = np.arange(drug_node_num, drug_node_num+relation_num).tolist()
    d_d_graph = nx.Graph(data_dd_list)
    j_j_graph = nx.Graph()
    d_j_graph = nx.Graph()
    d_j_graph.add_nodes_from(d_d_graph)
    for i in range(len(dis_node_list)):
        d_j_graph.add_node(dis_node_list[i])
        j_j_graph.add_node(dis_node_list[i])
    data_dj = pd.read_csv('../data/drug_disease.csv', names=columns, header=None)
    data_dj_list = data_dj.values.tolist()
    d_j_g = nx.Graph(data_dj_list)
    d_j_graph.add_edges_from(d_j_g.edges)
    j_j_g = nx.Graph(data_jj_list)
    j_j_graph.add_edges_from(j_j_g.edges)
    adj_d_j = torch.Tensor(nx.adjacency_matrix(d_j_graph).todense())
    adj_d_j = adj_d_j[:d_num, d_num:]
    adj_j_j = torch.Tensor(nx.adjacency_matrix(j_j_graph).todense())

    return adj_d_j, adj_j_j


def generate_train_graph(train_triplets):
    columns = ['Source', 'Target']
    data_dd = pd.read_csv('../data/drug_drug.csv', names=columns, header=None)
    data_dd_list = data_dd.values.tolist()
    graph = nx.Graph()
    d_d_graph = nx.Graph(data_dd_list)
    graph.add_nodes_from(d_d_graph)
    s = train_triplets[:, 0]
    o = train_triplets[:, 1]
    train_edges = np.stack([s, o], axis=1).tolist()
    d_d_tg = nx.Graph(train_edges)
    graph.add_edges_from(d_d_tg.edges)

    return graph
