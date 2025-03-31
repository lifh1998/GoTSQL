import re
import numpy as np
import torch


def generate_sample_graph_data(examples):
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


def generate_complete_graph_data(examples):
    node_names_list = []
    adj_matrix_list = []

    def set_one(matrix, i, j):
        matrix[i, j] = 1
        matrix[j, i] = 1

    for input_seq in examples['input_seq']:
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
            columns = [(col, dtype) for col, dtype in re.findall(
                r"\n\s*`?(\w+(?:\s+\w+)?)`?\s+(\w+)(?:,\s*--\s*.*?example:.*)?(?:,|$)",
                table_content
            ) if col.upper() not in ('PRIMARY KEY', 'CONSTRAINT')]
            column_dict = {}
            for column_name, data_type in columns:
                data_type_set.add(data_type.lower())
                column_dict[column_name.lower()] = data_type.lower()
            table_column_dict[table_name.lower()] = column_dict
            # 查找外键关系
            fks = re.findall(r"FOREIGN KEY \(`?(\w+)`?\) REFERENCES `?(\w+)`? \(`?(\w+)`?\)", table_content)
            for fk_column, ref_table, ref_column in fks:
                foreign_key_relationships.append(
                    (table_name.lower(), fk_column.lower(), ref_table.lower(), ref_column.lower()))

        # Create adjacency matrix
        num_nodes = len(node_names)
        num_tables = len(table_names)
        num_columns = sum(len(value) for value in table_column_dict.values())
        num_data_types = len(data_type_set)
        matrix_size = num_nodes + num_tables + num_columns + num_data_types
        adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
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
            if source_table in table_column_dict and target_table in table_column_dict:
                if source_fk in table_column_dict[source_table] and target_fk in table_column_dict[target_table]:
                    source_fk_index = table_column_dict[source_table][source_fk]
                    target_fk_index = table_column_dict[target_table][target_fk]
                    set_one(adj_matrix, source_fk_index, target_fk_index)
                else:
                    print(
                        f"可用的列: {list(table_column_dict[source_table].keys())} 和 {list(table_column_dict[target_table].keys())}")
                    raise ValueError(f"警告：找不到列 {source_table}: {source_fk} 或 {target_table}: {target_fk}")
            else:
                print(f"可用的表: {list(table_column_dict.keys())}")
                raise ValueError(f"警告：找不到表 {source_table} 或 {target_table}")
        adj_matrix_list.append(adj_matrix.tolist())

    result = {
        "input_seq": examples['input_seq'],
        "output_seq": examples['output_seq'],
        "node_names": node_names_list,
        "adj_matrix": adj_matrix_list,
    }
    return result


def process_seq_data(examples, tokenizer):
    # 对输入和响应进行 tokenization，启用 padding 和 truncation
    input_tokens = tokenizer(
        examples["input_seq"],
        padding=False,
        return_tensors="pt"
    )
    response_tokens = tokenizer(
        examples["output_seq"],
        padding=False,
        return_tensors="pt"
    )

    input_ids = input_tokens["input_ids"].clone().detach()
    attention_mask = input_tokens["attention_mask"].clone().detach()
    response_ids = response_tokens["input_ids"].clone().detach()
    response_mask = response_tokens["attention_mask"].clone().detach()

    # 拼接 input 和 response
    labels = torch.cat([
        torch.full((input_ids.shape[0], input_ids.shape[1]), -100, dtype=torch.long),
        response_ids
    ], dim=1)
    full_input_ids = torch.cat([input_ids, response_ids], dim=1)
    full_attention_mask = torch.cat([attention_mask, response_mask], dim=1)

    # 移除多余的维度（batch 维度）
    result = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "labels": labels,
        "node_names": examples["node_names"],
        "adj_matrix": examples["adj_matrix"]
    }
    return result


def process_graph_data(examples, tokenizer):
    # 处理图数据
    node_ids = [
        [tokenizer(node, return_tensors="pt", padding=False)["input_ids"].squeeze(0) for node in nodes]
        for nodes in examples["node_names"]
    ]

    return {
        "input_ids": examples["input_ids"],
        "attention_mask": examples["attention_mask"],
        "labels": examples["labels"],
        "got_nodes": node_ids,
        "adj_matrix": examples["adj_matrix"],
    }


def custom_data_collator(features, pad_token_id):
    batch = {}

    # Process sequence data with padding to max length in batch
    if "input_ids" in features[0]:
        # Get maximum length in the batch
        max_length = max(len(f["input_ids"]) for f in features)

        # Initialize tensors for padded data
        input_ids_padded = []
        attention_mask_padded = []
        labels_padded = []

        for f in features:
            # Current sequence length
            seq_len = len(f["input_ids"])

            # Pad input_ids with tokenizer.pad_token_id
            padded_input = torch.cat([
                f["input_ids"],
                torch.full((max_length - seq_len,), pad_token_id, dtype=torch.long)
            ])
            input_ids_padded.append(padded_input)

            # Pad attention_mask with zeros
            padded_mask = torch.cat([
                f["attention_mask"],
                torch.zeros(max_length - seq_len, dtype=torch.long)
            ])
            attention_mask_padded.append(padded_mask)

            # Pad labels with -100
            padded_labels = torch.cat([
                f["labels"],
                torch.full((max_length - seq_len,), -100, dtype=torch.long)
            ])
            labels_padded.append(padded_labels)

        # Stack tensors into batch
        batch["input_ids"] = torch.stack(input_ids_padded)
        batch["attention_mask"] = torch.stack(attention_mask_padded)
        batch["labels"] = torch.stack(labels_padded)

    # 特殊处理图结构数据
    if "got_nodes" in features[0]:
        # 找出该批次中的最大节点数和最大token长度
        max_nodes = max(f["got_nodes"].size(0) for f in features)
        max_node_len = max(f["got_nodes"].size(1) for f in features)

        # 创建填充后的张量
        padded_nodes = []
        for f in features:
            nodes = f["got_nodes"]
            n_nodes, n_len = nodes.size()
            padded = torch.zeros((max_nodes, max_node_len), dtype=nodes.dtype, device=nodes.device)
            padded[:n_nodes, :n_len] = nodes
            padded_nodes.append(padded)
        batch["got_nodes"] = torch.stack(padded_nodes)

    # 类似地处理邻接矩阵
    if "adj_matrix" in features[0]:
        max_size = max(f["adj_matrix"].size(0) for f in features)
        padded_adj = []
        for f in features:
            adj = f["adj_matrix"]
            n = adj.size(0)
            padded = torch.zeros((max_size, max_size), dtype=adj.dtype, device=adj.device)
            padded[:n, :n] = adj
            padded_adj.append(padded)
        batch["adj_matrix"] = torch.stack(padded_adj)

    return batch
