import pandas as pd
import numpy as np

# 重新生成数据，包括地理位置和设备类型，以满足后续转化为问答对、时序序列和知识图谱的需求

# 重新生成数据，确保所有数据集的时间戳完全对齐

# 生成性能管理数据
timestamp = pd.date_range(start="2021-01-01", periods=200, freq='H')
device_ids = np.random.randint(1, 20, size=200)


# 这里，pd.date_range 用于生成一个固定频率的时间序列。start="2021-01-01" 表示时间序列的开始日期，periods=200 表示生成200个时间点，
# freq='H' 表示时间频率为每小时。np.random.randint(1, 20, size=200) 生成200个介于1到19之间的随机整数，代表设备ID。

# 性能管理数据
rsrp = np.random.randint(-100, -50, size=200)
rsrq = np.random.randint(-20, -5, size=200)
sinr = np.random.randint(0, 30, size=200)
user_count = np.random.randint(50, 200, size=200)
traffic_data = np.random.randint(100, 1000, size=200)
latitude = np.random.uniform(30.0, 40.0, size=200)
longitude = np.random.uniform(120.0, 130.0, size=200)

# 这一部分代码使用NumPy来生成各种网络参数和地理位置信息：
# rsrp, rsrq, sinr 分别生成相关的无线信号质量参数。
# user_count 和 traffic_data 生成网络使用情况的数据。
# latitude 和 longitude 生成设备的随机地理位置坐标。

performance_data = pd.DataFrame({
    'timestamp': timestamp,
    'device_id': device_ids,
    'RSRP': rsrp,
    'RSRQ': rsrq,
    'SINR': sinr,
    'user_count': user_count,
    'traffic_data': traffic_data,
    'latitude': latitude,
    'longitude': longitude
})

# 这里创建了一个名为 performance_data 的DataFrame，列名为每个变量的名称。

# 故障管理数据
fault_types = ['网络中断', '信号质量下降', '硬件故障', '连接丢失']
fault_data = pd.DataFrame({
    'timestamp': timestamp,
    'device_id': device_ids,
    'fault_type': np.random.choice(fault_types, size=200),
    'recovery_time': np.random.randint(1, 60, size=200),
    'latitude': latitude,
    'longitude': longitude
})


# 这里生成故障类型和恢复时间。np.random.choice 用于从故障类型列表中随机选择故障，np.random.randint(1, 60, size=200) 生成恢复时间。

# 设备运维数据
operation_time = np.random.randint(900, 2000, size=200)
temperature = np.random.randint(20, 40, size=200)
device_type = np.random.choice(['Type A', 'Type B', 'Type C'], size=200)

maintenance_data = pd.DataFrame({
    'timestamp': timestamp,
    'device_id': device_ids,
    'operation_time': operation_time,
    'temperature': temperature,
    'device_type': device_type
})

# 生成设备的运行时间、温度和类型。与前面类似，数据被整合进一个DataFrame。

# 保存数据为CSV文件
performance_data.to_csv('data/performance_data_aligned.csv', index=False)
fault_data.to_csv('data/fault_data_aligned.csv', index=False, encoding='utf-8-sig')#改变乱码问题
maintenance_data.to_csv('data/maintenance_data_aligned.csv', index=False)

# 这些代码行将DataFrame保存为CSV文件，index=False 表示不保存行索引。
# 返回文件路径
# "/mnt/data/performance_data_aligned.csv", "/mnt/data/fault_data_aligned.csv", "/mnt/data/maintenance_data_aligned.csv"

