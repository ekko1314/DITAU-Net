import pandas as pd

def process_csv(input_file, output_file, labels_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 去掉第一行（索引0）和第一列
    df = df.iloc[1:, 1:]

    # 分离最后一列
    labels = df.iloc[:, -1]

    # 保存最后一列为新的独立CSV文件
    labels.to_csv(labels_file, index=False, header=False)

    # 去掉最后一列
    df = df.iloc[:, :-1]

    # 保存处理后的数据到新的文件
    df.to_csv(output_file, index=False, header=False)

# 输入和输出文件名
input_file = 'E:/Private_users/shengjiangtao/PythonObject/TimeSeriesAnomalyDetection/DTAAD-main/data/SWaT/test.csv'        # 输入CSV文件名
output_file = 'E:/Private_users/shengjiangtao/PythonObject/TimeSeriesAnomalyDetection/DTAAD-main/data/SWaT/test2.csv'  # 处理后的CSV文件名
labels_file = 'E:/Private_users/shengjiangtao/PythonObject/TimeSeriesAnomalyDetection/DTAAD-main/data/SWaT/labels.csv' # 输出的标签CSV文件名

# 调用函数处理CSV
process_csv(input_file, output_file, labels_file)

print("处理完成，已生成processed.csv和labels.csv文件。")
