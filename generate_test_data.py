import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_data(num_days=7):
    # 读取现有数据
    try:
        data = pd.read_csv('data.csv', encoding='gbk')
        
        # 确保日期列存在
        if '日期' not in data.columns:
            print('错误：数据文件中没有日期列')
            return
        
        # 转换日期列
        data['日期'] = pd.to_datetime(data['日期'])
        
        # 获取最后一行数据
        last_row = data.iloc[-1]
        last_date = pd.to_datetime(last_row['日期'])
        
        # 创建新数据列表
        new_data = []
        
        # 计算每列的统计特征
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        means = data[numeric_columns].mean()
        stds = data[numeric_columns].std()
        
        # 计算每列的趋势（使用最后30天数据）
        last_30_days = data.iloc[-30:]
        trends = {}
        for col in numeric_columns:
            if col != '日期':
                trend = np.polyfit(range(len(last_30_days)), last_30_days[col], deg=1)[0]
                trends[col] = trend
        
        # 生成新数据
        for i in range(1, num_days + 1):
            new_row = {}
            new_date = last_date + timedelta(days=i)
            new_row['日期'] = new_date.strftime('%Y-%m-%d')
            
            # 为每个数值列生成新的值
            for col in numeric_columns:
                if col != '日期':
                    # 基于均值、标准差和趋势生成新值
                    base_value = last_row[col]
                    trend_effect = trends[col] * i
                    random_effect = np.random.normal(0, stds[col] * 0.1)  # 使用较小的随机波动
                    new_value = base_value + trend_effect + random_effect
                    
                    # 确保生成的值为正数
                    new_value = max(new_value, 0)
                    
                    # 添加一些约束条件以保持数据合理性
                    if '比率' in col or '占比' in col:
                        new_value = min(new_value, 100)  # 比率不超过100%
                    
                    new_row[col] = round(new_value, 8)
            
            new_data.append(new_row)
        
        # 转换为DataFrame
        new_df = pd.DataFrame(new_data)
        
        # 追加到原始文件
        with open('data.csv', 'a', encoding='utf-8') as f:
            new_df.to_csv(f, header=False, index=False)
        
        print(f'成功生成并追加了{num_days}天的测试数据')
        print('新生成的数据日期范围：', new_data[0]['日期'], '至', new_data[-1]['日期'])
        
    except Exception as e:
        print(f'生成测试数据时出错：{str(e)}')

if __name__ == '__main__':
    # 默认生成7天的测试数据
    generate_test_data(14)