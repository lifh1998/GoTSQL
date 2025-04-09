import json
import re
import os


def extract_tables_from_sql(sql_query):
    """
    从SQL查询中提取表名
    """
    tables = set()

    # 匹配 FROM clause 中的表
    from_pattern = re.compile(r'FROM\s+([^\s,;()]+(?:\s*,\s*[^\s,;()]+)*)', re.IGNORECASE)
    from_matches = from_pattern.findall(sql_query)

    for match in from_matches:
        table_list = [t.strip() for t in match.split(',')]
        tables.update(table_list)

    # 匹配JOIN语句中的表
    join_pattern = re.compile(r'JOIN\s+([^\s,;()]+)', re.IGNORECASE)
    join_matches = join_pattern.findall(sql_query)
    tables.update(join_matches)

    return tables


def extract_tables_from_input(input_text):
    """
    从输入文本中提取表名，假设表名出现在CREATE TABLE语句中
    """
    create_table_pattern = re.compile(r'CREATE\s+TABLE\s+([^\s(]+)', re.IGNORECASE)
    return set(create_table_pattern.findall(input_text))


def process_json_file(input_file_path, output_file_path):
    """
    处理JSON文件，提取SQL语句中的表名并添加goal_table字段
    """
    try:
        # 读取输入JSON文件
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 处理每个条目
        for item in data:
            input_seq = item.get('input_seq', '')
            output_seq = item.get('output_seq', '')

            # 从output_seq中提取SQL代码块
            sql_pattern = re.compile(r'```sql\n(.*?)\n```', re.DOTALL)
            sql_matches = sql_pattern.findall(output_seq)

            all_tables = set()
            for sql in sql_matches:
                tables = extract_tables_from_sql(sql)
                all_tables.update(tables)

            # 从input_seq中提取有效表名
            valid_tables = extract_tables_from_input(input_seq)

            # 只保留在input_seq中存在的表
            final_tables = all_tables.intersection(valid_tables)

            # 添加goal_table字段
            item['goal_table'] = ','.join(sorted(final_tables))

        # 写入输出JSON文件
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        print(f"处理完成，结果已保存到 {output_file_path}")

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")


# 使用方法示例
if __name__ == "__main__":
    input_path = input("请输入JSON文件路径: ")
    output_dir = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    output_filename = "processed_" + filename
    output_path = os.path.join(output_dir, output_filename)

    process_json_file(input_path, output_path)