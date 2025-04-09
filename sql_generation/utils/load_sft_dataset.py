import json
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def find_sublist_index(lst, sublist):
    sublist_length = len(sublist)
    for i in range(len(lst) - sublist_length + 1):
        if lst[i:i + sublist_length] == sublist:
            return i
    return -1


def obtain_labels(input_ids, assistant_start_token_ids):
    '''
    Mask everything before assistant_start_token_ids with -100
    '''
    assistant_start_idx = find_sublist_index(input_ids, assistant_start_token_ids)
    if assistant_start_idx == -1:
        labels = input_ids
        print("length of the output sequence exceeds max length")
    else:
        labels = [-100] * assistant_start_idx + input_ids[assistant_start_idx:]
    assert len(input_ids) == len(labels)

    return labels


class SFTDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length, mode, use_graph=False, max_node_num=64, max_node_len=8):
        super().__init__()
        self.mode = mode
        self.use_graph = use_graph
        assistant_start_token_ids = [151644,
                                     77091]  # for Qwen2.5's tokenizer, the start token ids of the Assistant (<|im_start|>assistant)

        if mode == "pre-train":
            packed_data = np.load(data_dir)
            self.all_input_ids = torch.tensor(packed_data["all_packed_input_ids"], dtype=torch.int32)
            self.all_attention_mask = torch.tensor(packed_data["all_packed_attention_masks"], dtype=torch.int32)
            self.all_labels = torch.tensor(packed_data["all_packed_labels"], dtype=torch.int32)
            del packed_data
        elif mode == "sft":
            dataset = json.load(open(data_dir))
            sequences = [tokenizer.apply_chat_template([
                {"role": "user", "content": data["input_seq"]},
                {"role": "assistant", "content": data["output_seq"]}
            ], add_generation_prompt=False, tokenize=False) for data in tqdm(dataset)]

            tokenized_results = tokenizer.batch_encode_plus(
                sequences,
                truncation=False
            )

            self.all_input_ids = []
            self.all_attention_mask = []
            self.all_labels = []

            num = 0
            for input_ids in tokenized_results["input_ids"]:
                if len(input_ids) > max_length:  # pre-truncation
                    input_ids = input_ids[-max_length:]
                    num += 1
                self.all_input_ids.append(input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids)))
                self.all_attention_mask.append([1] * len(input_ids) + [0] * (max_length - len(input_ids)))
                # mask prompt loss
                self.all_labels.append(
                    obtain_labels(input_ids, assistant_start_token_ids) + [-100] * (max_length - len(input_ids)))
                # no-mask prompt loss
                # self.all_labels.append(input_ids + [-100] * (max_length-len(input_ids)))
            print(f"There are {num} sequences have been truncated.")

            self.all_input_ids = torch.tensor(self.all_input_ids, dtype=torch.int64)
            self.all_attention_mask = torch.tensor(self.all_attention_mask, dtype=torch.int64)
            self.all_labels = torch.tensor(self.all_labels, dtype=torch.int64)

            if use_graph:
                input_seqs = [data["input_seq"] for data in dataset]
                all_got_nodes, all_adj_matrix = self._extract_graph_data(input_seqs, tokenizer, max_node_num, max_node_len)
                self.all_got_nodes = torch.tensor(all_got_nodes, dtype=torch.int64)
                self.all_adj_matrix = torch.tensor(all_adj_matrix, dtype=torch.int64)

    def _extract_graph_data(self, input_seqs, tokenizer, max_node_num, max_node_len):
        node_names_list = []
        adj_matrix_list = []

        num=0
        for input_seq in tqdm(input_seqs):
            # 找到所有的 CREATE TABLE 语句
            tables = re.findall(r"CREATE TABLE `?(\w+)`?\s*\((.*?)\);", input_seq, re.DOTALL)
            if len(tables) > max_node_num:
                tables = tables[:max_node_num]
                num += 1
            table_names = [table_name for table_name, _ in tables]
            node_names_list.append(table_names)

            # Create adjacency matrix
            adj_matrix = np.zeros((max_node_num, max_node_num), dtype=int)
            for i in range(len(table_names)):
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
            adj_matrix_list.append(adj_matrix.tolist())
        print(f"There are {num} tables have been truncated.")

        # 处理图数据
        tokenized_results = tokenizer.batch_encode_plus(
            node_names_list,
            truncation=True,
            padding="max_length",
            max_length=max_node_len,
        )
        all_got_nodes = tokenized_results["input_ids"]
        if len(all_got_nodes) < max_node_num:
            zero_rows = [[0] * max_node_len for _ in range(max_node_num - len(all_got_nodes))]
            all_got_nodes.extend(zero_rows)
        all_adj_matrix = adj_matrix_list

        return all_got_nodes, all_adj_matrix

    def _extract_complete_graph_data(self, input_seqs, tokenizer, max_node_num, max_node_len):
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

    def __getitem__(self, index):
        if self.mode == "pre-train":
            return {
                "input_ids": torch.tensor(self.all_input_ids[index], dtype=torch.int64),
                "attention_mask": torch.tensor(self.all_attention_mask[index], dtype=torch.int64),
                "labels": torch.tensor(self.all_labels[index], dtype=torch.int64)
            }
        elif self.mode == "sft":
            sft_data = {
                "input_ids": self.all_input_ids[index],
                "attention_mask": self.all_attention_mask[index],
                "labels": self.all_labels[index]
            }
            if self.use_graph:
                sft_data["got_nodes"] = self.all_got_nodes[index]
                sft_data["adj_matrix"] = self.all_adj_matrix[index]
            return sft_data

    def __len__(self):
        return self.all_input_ids.shape[0]
