# 首先，我们加载已经对齐的数据
import pandas as pd

# 加载数据
performance_data = pd.read_csv('data/performance_data_aligned.csv')
fault_data = pd.read_csv('data/fault_data_aligned.csv')
maintenance_data = pd.read_csv('data/maintenance_data_aligned.csv')

# 这部分代码通过导入Pandas库，并使用 pd.read_csv() 方法加载存储在CSV文件中的对齐后的数据集。每个数据集分别包含性能管理、故障管理和设备运维的数据。


# 时序序列数据已经以时间戳对齐，可以直接使用或加工
time_series_data = performance_data.set_index('timestamp')
print("Time Series Data Sample:")
print(time_series_data.head())

# 这里将性能管理数据的时间戳设置为DataFrame的索引，方便后续的时序分析和处理。然后打印出前五条数据以供查看和验证。

# 生成问答对
qa_pairs = []
for _, row in fault_data.iterrows():
    question = f"在{row['timestamp']}，设备{row['device_id']}发生了什么故障？"
    answer = f"设备{row['device_id']}在{row['timestamp']}发生了{row['fault_type']}故障。"
    qa_pairs.append({'question': question, 'answer': answer})

qa_df = pd.DataFrame(qa_pairs)
print("\nQ&A Pairs Sample:")
print(qa_df.head())

# 这部分代码遍历故障数据中的每一行，构造关于设备故障的问答对，然后将这些问答对存储在一个新的DataFrame中。这些问答对可以用于训练NLP模型或进行自然语言处理任务。

# 构建知识图谱
import networkx as nx

G = nx.Graph()
# 添加设备节点
for _, row in maintenance_data.iterrows():
    G.add_node(row['device_id'], type='Device', operation_time=row['operation_time'], device_type=row['device_type'])

# 添加故障事件节点和边
for _, row in fault_data.iterrows():
    G.add_node(row['timestamp'], type='Fault', fault_type=row['fault_type'])
    G.add_edge(row['device_id'], row['timestamp'], relation='Occurred')

print("\nKnowledge Graph Info:")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")


# 在这一部分，使用NetworkX库构建一个知识图谱，图中节点代表设备和故障事件，边代表故障发生的关系。通过添加设备和故障类型作为节点，
# 再通过设备和相应的故障时间建立边，形成了一个表示故障关系的图结构。

# 数据转化已完成，下面可以进一步加工或保存处理结果

import os

# 确保目标文件夹存在
os.makedirs('data/datatransfer', exist_ok=True)

# 保存数据到指定的文件夹 'datatransfer' 下

# 保存时序序列数据
time_series_data.to_csv('data/datatransfer/time_series_data.csv')

# 保存问答对数据
qa_df.to_csv('data/datatransfer/qa_data.csv')

# 保存知识图谱数据，NetworkX图不能直接保存为CSV，所以我们先保存节点和边信息为CSV
nodes_data = pd.DataFrame(list(G.nodes(data=True)), columns=['Node', 'Attributes'])
edges_data = pd.DataFrame(list(G.edges(data=True)), columns=['Source', 'Target', 'Attributes'])


# 保存知识图谱数据，包括节点和边信息
nodes_data.to_csv('data/datatransfer/knowledge_graph_nodes.csv', index=False)
edges_data.to_csv('data/datatransfer/knowledge_graph_edges.csv', index=False)


# 最后这部分代码首先确保保存数据的目标文件夹存在，然后将处理好的时序数据、问答对数据以及知识图谱的节点和边信息保存为CSV文件。

# 提供文件路径
'/mnt/data/datatransfer/time_series_data.csv', '/mnt/data/datatransfer/qa_data.csv', '/mnt/data/datatransfer/knowledge_graph_nodes.csv', '/mnt/data/datatransfer/knowledge_graph_edges.csv'
