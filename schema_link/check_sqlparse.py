import json


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def compare_goal_tables(json1, json2):
    mismatched_indices = []

    for index, (entry1, entry2) in enumerate(zip(json1, json2)):
        tables1 = set(entry1['goal_table'].split(','))
        tables2 = set(entry2['goal_table'].split(','))

        if tables1 != tables2:
            mismatched_indices.append(index)

    return mismatched_indices


def main():
    # 读取两个JSON文件
    json_file_1 = 'D:\\ujs\workspace\experiments\GoTSQL\data\processed_2_train_synsql_sample.json'  # 替换为你的第一个JSON文件路径
    json_file_2 = 'D:\\ujs\workspace\experiments\GoTSQL\data\processed_train_synsql_sample.json'  # 替换为你的第二个JSON文件路径

    json_data_1 = load_json(json_file_1)
    json_data_2 = load_json(json_file_2)

    # 比较goal_table
    mismatches = compare_goal_tables(json_data_1, json_data_2)

    if mismatches:
        print(f"不一致的索引位置: {mismatches}")
    else:
        print("所有索引位置的goal_table一致。")


if __name__ == "__main__":
    main()