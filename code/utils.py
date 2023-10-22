import pandas
import torch

from datetime import datetime


def load_data_from_csv(train_node_path='../data/train_90.csv', train_edge_path='../data/edge_90.csv',
                       test_node_path='../data/test_A/node_test_4_A.csv',
                       test_edge_path='../data/test_A/edge_test_4_A.csv'):
    data_train_node = pandas.read_csv(train_node_path)
    data_train_edge = pandas.read_csv(train_edge_path)
    data_test_node = pandas.read_csv(test_node_path)
    data_test_edge = pandas.read_csv(test_edge_path)

    return data_train_node, data_train_edge, data_test_node, data_test_edge


def map_date_to_index(date, base_date_str='20230104'):
    date_str = str(date)
    try:
        # 将日期字符串转换为日期对象
        date_format = "%Y%m%d"
        date_obj = datetime.strptime(date_str, date_format)
        base_date_obj = datetime.strptime(base_date_str, date_format)
    except ValueError:
        return None
    else:
        # 计算日期差值并映射到从0开始的数字
        delta = (date_obj - base_date_obj).days
        assert delta >= 0, f"{base_date_str}, 不是数据集中出现的最小日期"
        mapped_index = max(delta, 0)
        return mapped_index


def load_node_data(data_node, X, Y, num_node, date_start, date_end, 
                   feature_start, feature_end, num_target, dict_geohash_to_id):
    num_date = date_end - date_start
    num_feature = feature_end - feature_start
    for index_node, index_row in enumerate(range(0, num_node * num_date, num_date)):
        geohash_id = data_node['geohash_id'].iloc[index_row]
        assert geohash_id in dict_geohash_to_id, f'{geohash_id} is not found.'
        assert index_row + num_date <= len(data_node), f'Index {index_row + num_date} overbounds'
        node_id = dict_geohash_to_id[geohash_id]
        
        index_col = data_node.columns.get_loc('F_1')
        node_features = data_node.iloc[index_row:index_row+num_date, index_col:index_col+num_feature].values
        X[node_id, date_start:date_end, feature_start:feature_end] = torch.tensor(node_features)

        # 若data_node为data_test_node，则其不含特征值，即num_target=0，跳过即可
        if num_target == 0: continue
        index_col = data_node.columns.get_loc('active_index')
        node_targets = data_node.iloc[index_row:index_row+num_date, index_col:index_col+num_target].values
        Y[node_id, date_start:date_end, :] = torch.tensor(node_targets)


def group_clear_data(data_edge, X, feature_start, feature_end, dict_geohash_to_id, groupby):
    num_feature = feature_end - feature_start

    data_edge_grouped = data_edge.groupby([groupby, 'date_id']).agg({'F_1': 'sum', 'F_2': 'sum'})
    data_edge_grouped = data_edge_grouped.reset_index()
    # print(data_edge_grouped)

    data_edge_grouped['node_index'] = data_edge_grouped[groupby].apply(lambda x: dict_geohash_to_id.get(x, None))
    data_edge_grouped['date_index'] = data_edge_grouped['date_id'].apply(lambda x: map_date_to_index(x))

    data_edge_grouped_cleared = data_edge_grouped.dropna(subset=['node_index', 'date_index'])
    
    node_indices = data_edge_grouped_cleared['node_index'].values.astype('int64')
    date_indices = data_edge_grouped_cleared['date_index'].values.astype('int64')
    # print(node_indices)

    index_col_F1 = data_edge_grouped_cleared.columns.get_loc('F_1')
    edge_features = data_edge_grouped_cleared.iloc[:, index_col_F1:index_col_F1+num_feature].values
    edge_features = torch.tensor(edge_features).to(torch.float)

    return node_indices, date_indices, edge_features


def load_edge_data(data_edge, X, feature_start, feature_end, dict_geohash_to_id):
    # 处理每条边的第一个节点
    node_indices, date_indices, edge_features = group_clear_data(data_edge, X, feature_start, feature_end, 
                                                                 dict_geohash_to_id, 'geohash6_point1')
    X[node_indices, date_indices, feature_start:feature_end] += edge_features
    
    # 处理每条边的第二个节点
    node_indices, date_indices, edge_features = group_clear_data(data_edge, X, feature_start, feature_end, 
                                                                 dict_geohash_to_id, 'geohash6_point2')
    edge_features[:, [0, 1]] = edge_features[:, [1, 0]].clone()
    X[node_indices, date_indices, feature_start:feature_end] += edge_features


def load_data():
    # 从csv文件中，读取训练数据集和测试数据集的节点信息、边信息
    data_train_node, data_train_edge, data_test_node, data_test_edge = load_data_from_csv()

    # 创建geohash_id到node_id的映射字典
    dict_geohash_to_id = {} 
    list_geohash = data_train_node['geohash_id']
    for geohash in list_geohash:
            if geohash not in dict_geohash_to_id:
                dict_geohash_to_id[geohash] = len(dict_geohash_to_id)

    # 节点数量
    num_node = len(dict_geohash_to_id)

    # 日期数量
    num_date_train = 90
    num_date_test = 4
    num_date = num_date_train + num_date_test

    # 特征数量
    num_feature_node = 35
    num_feature_edge = 2
    num_feature = num_feature_node + num_feature_edge

    # 目标数量
    num_target = 2

    # 输入张量，输出张量
    X = torch.zeros(num_node, num_date, num_feature)
    Y = torch.zeros(num_node, num_date, num_target)

    # 读取data_train_node，data_test_node，更新X，Y
    load_node_data(data_train_node, X, Y, num_node, 0, num_date_train, 
               0, num_feature_node, num_target, dict_geohash_to_id)
    load_node_data(data_test_node, X, Y, num_node, num_date_train, num_date, 
                   0, num_feature_node, 0, dict_geohash_to_id)
    
    # 读取data_train_edge，data_test_edge，更新X
    load_edge_data(data_train_edge, X, num_feature_node, num_feature, dict_geohash_to_id)
    load_edge_data(data_test_edge, X, num_feature_node, num_feature, dict_geohash_to_id)

    # X shape(num_node, num_date, num_feature)
    # Y shape(num_node, num_date, num_target)
    return X, Y  