import numpy as np
import pandas as pd

# 设置随机种子以确保结果可重复
np.random.seed(42)


# 定义数据生成函数
def generate_data(num_samples):
    data = {
        '深度': np.random.randint(500, 1500, num_samples),  # 深度范围：500m到1500m
        '应力': np.random.uniform(10, 35, num_samples),  # 应力范围：10MPa到35MPa
        '倾角': np.random.randint(0, 90, num_samples),  # 倾角范围：0°到90°
        '倾向': np.random.randint(0, 360, num_samples)  # 倾向范围：0°到360°
    }
    return pd.DataFrame(data)


# 生成1000条模拟数据
num_samples = 1000
df = generate_data(num_samples)

# 将数据保存为CSV文件
df.to_csv('HTPF.csv', index=False, encoding='utf-8-sig')
print('模拟数据已生成并保存为CSV文件。')
