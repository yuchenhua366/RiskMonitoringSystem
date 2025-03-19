import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建图片保存目录
IMAGE_DIR = 'analysis_images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def load_data(file_path):
    """加载数据文件，支持.csv、.xlsx和.xls格式"""
    try:
        # 获取文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        df = None
        
        # 根据文件扩展名选择不同的读取方法
        if file_ext == '.csv':
            # 尝试不同的编码方式读取CSV文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    if not df.columns[0].startswith('�'):
                        break
                except UnicodeDecodeError:
                    continue
        elif file_ext in ['.xlsx', '.xls']:
            try:
                # 读取Excel文件，默认读取第一个sheet
                df = pd.read_excel(file_path)
            except Exception as excel_e:
                raise ValueError(f'读取Excel文件时出错：{str(excel_e)}')
        else:
            raise ValueError(f'不支持的文件格式：{file_ext}')
        
        if df is None:
            raise ValueError(f'无法读取文件：{file_path}')
        
        # 检查日期列是否存在并进行转换
        if '日期' in df.columns:
            try:
                # 尝试将日期列转换为datetime格式
                df['日期'] = pd.to_datetime(df['日期'], format='%Y-%m-%d')
            except Exception:
                try:
                    # 如果指定格式转换失败，尝试自动识别格式
                    df['日期'] = pd.to_datetime(df['日期'])
                except Exception as e:
                    print(f'日期转换出错：{str(e)}')
        
        # 处理百分比格式的数据
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # 检查是否为百分比格式
                    if df[col].str.contains('%').any():
                        # 移除百分号并转换为浮点数
                        df[col] = df[col].str.rstrip('%').astype('float') / 100
                except:
                    continue
        
        return df
    except Exception as e:
        raise ValueError(f'读取数据文件时出错：{str(e)}')

def basic_info_analysis(df):
    """基础信息分析
    包括数据维度、字段类型、缺失值等基本信息，并处理异常值、缺失值和无穷值
    """
    # 设置pandas显示选项，确保所有列都能完整显示
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)  # 设置足够大的宽度
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)  # 禁止换行显示
    
    print('数据基本信息：')
    print(f'数据维度: {df.shape}')
    print('\n字段类型信息：')
    print(df.dtypes.to_frame().to_string())
    print('\n缺失值信息：')
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_info = pd.DataFrame({'缺失值数量': missing, '缺失比例(%)': missing_percent.round(2)})
    print(missing_info[missing_info['缺失值数量'] > 0].to_string())
    
    # 处理无穷值，使用更新的方法
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 处理缺失值
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 数值型变量使用中位数填充
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    
    # 类别型变量使用众数填充
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
    
    # 处理异常值
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # 将异常值替换为边界值
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 重置pandas显示选项，使用更新的方法
    pd.reset_option('display')
    
    return df

def distribution_analysis(df):
    """数据分布分析
    包括数值型变量的统计特征和分布可视化
    """
    # 数值型变量统计特征
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print('数值型变量统计特征：')
    print(df[numeric_cols].describe())
    
    # 分布可视化
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'{col}分布直方图')
        plt.subplot(122)
        stats.probplot(df[col].dropna(), dist="norm", plot=plt)
        plt.title(f'{col} Q-Q图')
        plt.tight_layout()
        # 保存图片，替换文件名中的特殊字符
        safe_col_name = col.replace('/', '_').replace('\\', '_')
        plt.savefig(os.path.join(IMAGE_DIR, f'{safe_col_name}_distribution.png'))
        plt.close()

def correlation_analysis(df):
    """相关性分析
    计算数值型变量间的相关系数并可视化，只展示相关性最强的8个特征
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr()
    
    # 获取相关系数矩阵的上三角部分（不包括对角线）
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 将相关系数展平并按绝对值排序
    sorted_pairs = []
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            if not pd.isna(upper_tri.loc[idx, col]):
                sorted_pairs.append((idx, col, abs(upper_tri.loc[idx, col])))
    
    sorted_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # 获取相关性最强的8个特征对
    top_features = set()
    for pair in sorted_pairs:
        top_features.add(pair[0])
        top_features.add(pair[1])
        if len(top_features) >= 8:
            break
    
    # 筛选相关性最强的特征
    top_features = list(top_features)[:8]
    filtered_corr = corr_matrix.loc[top_features, top_features]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(filtered_corr, 
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .5})
    
    plt.title('主要特征相关性热力图', fontsize=12, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 调整布局以防止标签被切割
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(IMAGE_DIR, 'correlation_heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close()

def categorical_analysis(df):
    """类别型变量分析
    统计类别型变量的分布情况
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f'\n{col}的分布情况：')
        value_counts = df[col].value_counts()
        print(value_counts)
        print(f'占比(%)：')
        print((value_counts / len(df) * 100).round(2))

def outlier_detection(df):
    """异常值检测
    使用箱线图方法检测数值型变量的异常值
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            print(f'\n{col}的异常值情况：')
            print(f'异常值数量：{len(outliers)}')
            print(f'异常值占比：{(len(outliers)/len(df)*100):.2f}%')
            print(f'异常值范围：<{lower_bound:.2f}或>{upper_bound:.2f}')

def time_series_analysis(df, time_col):
    """时间序列分析
    分析时间相关的趋势和模式
    """
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        
        # 时间序列可视化
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df[col])
            plt.title(f'{col}随时间的变化趋势')
            plt.xticks(rotation=45)
            plt.tight_layout()
            # 保存图片
            plt.savefig(os.path.join(IMAGE_DIR, f'{col}_time_series.png'))
            plt.close()

def detect_data_anomalies(df):
    """检测数据异常
    检测并可视化数据中的异常模式，包括离群值、突变点和不合理的数值范围
    """
    anomaly_results = {}
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        # 计算Z-score
        z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
        # 识别异常值（Z-score > 3）
        anomalies = df[z_scores > 3]
        
        if len(anomalies) > 0:
            # 记录异常值信息
            anomaly_results[col] = {
                '异常值数量': len(anomalies),
                '异常值比例': f'{(len(anomalies) / len(df) * 100):.2f}%',
                '异常值范围': f'[{anomalies[col].min():.2f}, {anomalies[col].max():.2f}]'
            }
            
            # 创建箱线图和散点图
            plt.figure(figsize=(12, 5))
            
            # 箱线图
            plt.subplot(121)
            sns.boxplot(y=df[col])
            plt.title(f'{col}的箱线图分析')
            
            # 散点图（带异常值标记）
            plt.subplot(122)
            plt.scatter(range(len(df)), df[col], c='blue', alpha=0.5, label='正常值')
            plt.scatter(anomalies.index, anomalies[col], c='red', alpha=0.7, label='异常值')
            plt.title(f'{col}的异常值分布')
            plt.legend()
            
            plt.tight_layout()
            safe_col_name = col.replace('/', '_').replace('\\', '_')
            plt.savefig(os.path.join(IMAGE_DIR, f'{safe_col_name}_anomalies.png'))
            plt.close()
    
    return anomaly_results

def generate_report(df):
    """生成数据分析报告
    将所有分析结果和图片整合到一个HTML报告中
    """
    import io
    from datetime import datetime
    
    # 创建一个StringIO对象来捕获print输出
    output = io.StringIO()
    import sys
    sys.stdout = output
    
    # 执行各项分析
    basic_info_analysis(df)
    distribution_analysis(df)
    correlation_analysis(df)
    categorical_analysis(df)
    outlier_detection(df)
    
    # 执行异常数据检测
    anomaly_results = detect_data_anomalies(df)
    
    # 恢复标准输出
    sys.stdout = sys.__stdout__
    analysis_output = output.getvalue()
    output.close()
    
    # 获取数值型列名
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # 生成异常检测结果HTML表格
    anomaly_table = '<h2>4. 异常数据检测结果</h2>'
    if anomaly_results:
        anomaly_table += '''
        <table border="1" style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr style="background-color: #f2f2f2;">
                <th style="padding: 12px;">特征</th>
                <th style="padding: 12px;">异常值数量</th>
                <th style="padding: 12px;">异常值比例</th>
                <th style="padding: 12px;">异常值范围</th>
            </tr>
        '''
        for col, info in anomaly_results.items():
            anomaly_table += f'''
            <tr>
                <td style="padding: 8px;">{col}</td>
                <td style="padding: 8px;">{info['异常值数量']}</td>
                <td style="padding: 8px;">{info['异常值比例']}</td>
                <td style="padding: 8px;">{info['异常值范围']}</td>
            </tr>
            '''
        anomaly_table += '</table>'
        
        # 添加异常值分布图
        anomaly_table += '<div class="anomaly-plots">'
        for col in anomaly_results.keys():
            safe_col_name = col.replace('/', '_').replace('\\', '_')
            anomaly_table += f'''
            <div style="margin: 20px 0;">
                <h3>{col}的异常值分析</h3>
                <img src="{IMAGE_DIR}/{safe_col_name}_anomalies.png" alt="{col}的异常值分析图" style="max-width: 100%; height: auto;">
            </div>
            '''
        anomaly_table += '</div>'
    else:
        anomaly_table += '<p>未检测到显著的异常数据。</p>'
    
    # 生成HTML报告
    report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>数据分析报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            h3 {{ color: #7f8c8d; margin-top: 20px; }}
            pre {{ 
                background-color: #f8f9fa; 
                padding: 15px; 
                border-radius: 5px; 
                white-space: pre-wrap;
                word-wrap: break-word;
                max-width: 100%;
                overflow-x: auto;
                font-size: 14px;
                line-height: 1.5;
            }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #eee; border-radius: 5px; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            .distribution-section {{ margin-bottom: 40px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f5f5f5; }}
            .stats-table {{ margin-top: 20px; }}
            .stats-table th {{ background-color: #f5f5f5; font-weight: bold; }}
            .stats-table td {{ text-align: right; }}
            .anomaly-section {{ margin-top: 40px; }}
            .anomaly-plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        </style>
    </head>
    <body>
        <h1>数据分析报告</h1>
        <p class="timestamp">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>1. 数据基本信息</h2>
        <div class="basic-info">
            <pre>{analysis_output}</pre>
            
            <h3>数值型变量统计特征</h3>
            <table class="stats-table">
                <tr>
                    <th>统计量</th>
                    {''.join([f'<th>{col}</th>' for col in numeric_cols])}
                </tr>
                <tr>
                    <td>样本数</td>
                    {''.join([f'<td>{df[col].count()}</td>' for col in numeric_cols])}
                </tr>
                <tr>
                    <td>均值</td>
                    {''.join([f'<td>{df[col].mean():.4f}</td>' for col in numeric_cols])}
                </tr>
                <tr>
                    <td>标准差</td>
                    {''.join([f'<td>{df[col].std():.4f}</td>' for col in numeric_cols])}
                </tr>
                <tr>
                    <td>最小值</td>
                    {''.join([
                        f'<td style="color: {"red" if abs(df[col].min() - df[col].mean()) > 2 * df[col].std() else "black"}">{df[col].min():.4f}</td>'
                        for col in numeric_cols
                    ])}
                </tr>
                <tr>
                    <td>25%分位数</td>
                    {''.join([
                        f'<td style="color: {"red" if abs(df[col].quantile(0.25) - df[col].mean()) > 2 * df[col].std() else "black"}">{df[col].quantile(0.25):.4f}</td>'
                        for col in numeric_cols
                    ])}
                </tr>
                <tr>
                    <td>中位数</td>
                    {''.join([
                        f'<td style="color: {"red" if abs(df[col].median() - df[col].mean()) > 2 * df[col].std() else "black"}">{df[col].median():.4f}</td>'
                        for col in numeric_cols
                    ])}
                </tr>
                <tr>
                    <td>75%分位数</td>
                    {''.join([
                        f'<td style="color: {"red" if abs(df[col].quantile(0.75) - df[col].mean()) > 2 * df[col].std() else "black"}">{df[col].quantile(0.75):.4f}</td>'
                        for col in numeric_cols
                    ])}
                </tr>
                <tr>
                    <td>最大值</td>
                    {''.join([
                        f'<td style="color: {"red" if abs(df[col].max() - df[col].mean()) > 2 * df[col].std() else "black"}">{df[col].max():.4f}</td>'
                        for col in numeric_cols
                    ])}
                </tr>
            </table>
        </div>
        
        <h2>2. 数据分布可视化</h2>
        <div class="distribution-section">
        {''.join([f'<h3>{col}的分布分析</h3><img src="{IMAGE_DIR}/{col.replace("/", "_").replace("\\", "_")}_distribution.png" alt="{col}的分布图">' for col in numeric_cols])}
        </div>
        
        <h2>3. 相关性分析</h2>
        <img src="{IMAGE_DIR}/correlation_heatmap.png" alt="相关性热力图">
        
        <div class="anomaly-section">
            {anomaly_table}
        </div>
    </body>
    </html>
    """
    
    # 保存报告
    report_path = 'analysis_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f'分析报告已生成：{report_path}')

def main(file_path='data.csv'):
    # 加载数据
    df = load_data(file_path)
    
    # 执行各项分析
    basic_info_analysis(df)
    distribution_analysis(df)
    correlation_analysis(df)
    categorical_analysis(df)
    outlier_detection(df)
    
    # 生成分析报告
    generate_report(df)
    
    # 如果数据包含时间字段，可以进行时间序列分析
    # time_series_analysis(df, 'time_column')
    
    print(f'分析完成！图片已保存在 {IMAGE_DIR} 目录下。')

if __name__ == '__main__':
    main()