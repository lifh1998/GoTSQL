from datasets import Dataset

from process_dataset import generate_sample_graph_data, generate_graph_data


def test_extract_graph_without_column(dataset):
    return dataset.map(generate_sample_graph_data, batched=True)


def test_extract_graph_with_column(dataset):
    return dataset.map(generate_graph_data, batched=True)


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
