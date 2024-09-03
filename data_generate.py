import pandas as pd
import numpy as np

# 生成性能管理数据
timestamp = pd.date_range(start="2021-01-01", periods=200, freq='H')
rsrp = np.random.randint(-100, -50, size=200)
rsrq = np.random.randint(-20, -5, size=200)
sinr = np.random.randint(0, 30, size=200)
user_count = np.random.randint(50, 200, size=200)
traffic_data = np.random.randint(100, 1000, size=200)

performance_data = pd.DataFrame({
    'timestamp': timestamp,
    'RSRP': rsrp,
    'RSRQ': rsrq,
    'SINR': sinr,
    'user_count': user_count,
    'traffic_data': traffic_data
})

# 生成故障管理数据
fault_types = ['网络中断', '信号质量下降', '硬件故障', '连接丢失']
fault_data = pd.DataFrame({
    'timestamp': pd.date_range(start="2021-01-01 00:30", periods=200, freq='H'),
    'fault_type': np.random.choice(fault_types, size=200),
    'recovery_time': np.random.randint(1, 60, size=200)
})

# 生成设备运维数据
device_ids = np.random.randint(1, 10, size=200)
operation_time = np.random.randint(900, 2000, size=200)
temperature = np.random.randint(20, 40, size=200)

maintenance_data = pd.DataFrame({
    'timestamp': pd.date_range(start="2021-01-01", periods=200, freq='H'),
    'device_id': device_ids,
    'operation_time': operation_time,
    'temperature': temperature
})

# 保存数据为CSV文件
performance_data.to_csv('data/performance_data.csv', index=False)
fault_data.to_csv('data/fault_data.csv', index=False)
maintenance_data.to_csv('data/maintenance_data.csv', index=False)

# 返回文件路径
"/mnt/data/performance_data.csv", "/mnt/data/fault_data.csv", "/mnt/data/maintenance_data.csv"

