import re
from datasets import Dataset
import numpy as np


def test_extract_graph_without_column(dataset):
    def augment_data(examples):
        node_names_list = []
        adj_matrices_list = []

        for input_seq in examples['input_seq']:
            # 找到所有的 CREATE TABLE 语句
            tables = re.findall(r"CREATE TABLE `?(\w+)`?\s*\((.*?)\);", input_seq, re.DOTALL)
            table_names = [table_name for table_name, _ in tables]
            node_names_list.append(table_names)

            # Create adjacency matrix
            num_tables = len(table_names)
            adj_matrix = np.zeros((num_tables, num_tables), dtype=int)
            for i in range(num_tables):
                adj_matrix[i, i] = 1

            foreign_key_relationships = []
            # 遍历每个表的定义
            for table_name, table_content in tables:
                # 查找外键关系
                fks = re.findall(r"FOREIGN KEY \(`?(\w+)`?\) REFERENCES `?(\w+)`? \(`?(\w+)`?\)", table_content)
                for fk_column, ref_table, ref_column in fks:
                    foreign_key_relationships.append((table_name, fk_column, ref_table, ref_column))

            for relationship in foreign_key_relationships:
                source_table = relationship[0]
                source_fk = relationship[1]
                target_table = relationship[2]
                target_fk = relationship[3]
                print(source_table, source_fk, target_table, target_fk)
                if source_table in table_names and target_table in table_names:
                    source_index = table_names.index(source_table)
                    target_index = table_names.index(target_table)
                    adj_matrix[source_index, target_index] = 1
                    adj_matrix[target_index, source_index] = 1
            adj_matrices_list.append(adj_matrix.tolist())

        result = {
            "input_seq": examples['input_seq'],
            "output_seq": examples['output_seq'],
            "node_names": node_names_list,
            "adj_matrices": adj_matrices_list,
        }
        return result

    dataset = dataset.map(augment_data, batched=True)
    return dataset


def test_extract_graph_with_column(dataset):
    def set_one(matrix, i, j):
        matrix[i, j] = 1
        matrix[j, i] = 1

    def augment_data(examples):
        node_names_list = []
        adj_matrices_list = []

        for input_seq in examples['input_seq']:
            print(input_seq)
            node_names = ['TABLE', 'COLUMN', 'DATA_TYPE']

            # 找到所有的 CREATE TABLE 语句
            tables = re.findall(r"CREATE TABLE `?(\w+)`?\s*\((.*?)\);", input_seq, re.DOTALL)
            table_names = [table_name.lower() for table_name, _ in tables]

            foreign_key_relationships = []
            table_column_dict = {}
            data_type_set = set()
            # 遍历每个表的定义
            for table_name, table_content in tables:
                # 找到所有的列
                columns = re.findall(r"\n\s*`?(\w+(?:\s+\w+)?)`?\s+(\w+),\s*--\s*.*?example:", table_content)
                column_dict = {}
                for column_name, data_type in columns:
                    data_type_set.add(data_type.lower())
                    column_dict[column_name.lower()] = data_type.lower()
                table_column_dict[table_name.lower()] = column_dict
                # 查找外键关系
                fks = re.findall(r"FOREIGN KEY \(`?(\w+)`?\) REFERENCES `?(\w+)`? \(`?(\w+)`?\)", table_content)
                for fk_column, ref_table, ref_column in fks:
                    foreign_key_relationships.append((table_name.lower(), fk_column.lower(), ref_table.lower(), ref_column.lower()))

            # Create adjacency matrix
            num_nodes = len(node_names)
            num_tables = len(table_names)
            num_columns = sum(len(value) for value in table_column_dict.values())
            num_data_types = len(data_type_set)
            matrix_size  = num_nodes + num_tables + num_columns + num_data_types
            adj_matrix = np.zeros((matrix_size , matrix_size ), dtype=int)
            for i in range(matrix_size):
                adj_matrix[i, i] = 1

            # 处理结点以及关系
            for data_type in data_type_set:
                node_names.append(data_type)
                set_one(adj_matrix, 2, len(node_names) - 1)
            for table_name in table_names:
                node_names.append(table_name)
                set_one(adj_matrix, 0, len(node_names) - 1)
                for column_name, data_type in table_column_dict[table_name].items():
                    node_names.append(column_name)
                    curr_index = len(node_names) - 1
                    set_one(adj_matrix, 1, curr_index)
                    set_one(adj_matrix, node_names.index(table_name), curr_index)
                    set_one(adj_matrix, node_names.index(data_type), curr_index)
                    # 替换数据，方便找外键关系
                    table_column_dict[table_name][column_name] = curr_index
            node_names_list.append(node_names)

            # 处理外键关系
            for relationship in foreign_key_relationships:
                source_table = relationship[0]
                source_fk = relationship[1]
                target_table = relationship[2]
                target_fk = relationship[3]
                print(source_table, source_fk, target_table, target_fk)
                if source_table.lower() in node_names and target_table.lower() in node_names:
                    source_fk_index = table_column_dict[source_table][source_fk]
                    target_fk_index = table_column_dict[target_table][target_fk]
                    set_one(adj_matrix, source_fk_index, target_fk_index)
            adj_matrices_list.append(adj_matrix.tolist())

        result = {
            "input_seq": examples['input_seq'],
            "output_seq": examples['output_seq'],
            "node_names": node_names_list,
            "adj_matrices": adj_matrices_list,
        }
        return result

    dataset = dataset.map(augment_data, batched=True)
    return dataset


if __name__ == "__main__":
    dataset = (Dataset.from_json("D:\\ujs\\workspace\\datasets\\OmniSQL\\data\\train_spider.json")
               .shuffle(seed=42)
               .select(range(5)))

    dataset = test_extract_graph_with_column(dataset)

    for data in dataset:
        # print(data['input_seq'])
        # print(data['output_seq'])
        print(f"结点列表：{data['node_names']}")
        print(f"邻接矩阵：{data['adj_matrices']}")
