import pandas as pd
import numpy as np
from datetime import datetime
import os
import argparse
import json

# 导入自定义模块
from data_preprocessing import load_data, basic_info_analysis, distribution_analysis, correlation_analysis, generate_report
from feature_engineering import FeatureEngineering
from feature_selection import FeatureSelection

def load_config(config_path='config.json'):
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'data_file': 'data.csv',
        'column_mapping': {
            '贷款余额限制': '贷款余额/存款余额',
            '贷款余额': '贷款余额',
            '存款余额': '存款余额',
            '票据承兑余额': '票据承兑余额',
            '存放同业余额': '存放同业余额'
        }
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='风险监测系统')
    parser.add_argument('--data', type=str, help='数据文件路径')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    data_file = args.data or config.get('data_file')
    
    print('开始执行风险监测流程...')
    print(f'使用数据文件: {data_file}')
    
    # 步骤1：数据预处理
    print('\n1. 数据预处理阶段')
    try:
        # 加载原始数据
        data = load_data(data_file)
        
        # 从配置文件获取列映射
        column_mapping = config.get('column_mapping', {})
        
        # 重命名列
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data[new_col] = data[old_col]
                
        # 确保'贷款余额/存款余额'列存在
        if '贷款余额/存款余额' not in data.columns and '贷款余额' in data.columns and '存款余额' in data.columns:
            data['贷款余额/存款余额'] = data['贷款余额'] / data['存款余额']
        
        # 数据探索性分析
        basic_info_analysis(data)
        distribution_analysis(data)
        correlation_analysis(data)
        
        # 生成数据分析报告
        generate_report(data)
        
        print('数据预处理完成')
        
    except Exception as e:
        print(f'数据预处理阶段出错：{str(e)}')
        return
    
    # 步骤2：特征工程
    print('\n2. 特征工程阶段')
    try:
        # 特征工程
        feature_engineering = FeatureEngineering()
        train_data,test_data = feature_engineering.process(data)
        print('特征工程完成')
        
    except Exception as e:
        print(f'特征工程阶段出错：{str(e)}')
        return
    
    # 步骤3：特征筛选
    print('\n3. 特征筛选阶段')
    try:
        # 创建特征选择实例
        feature_selector = FeatureSelection()
        
        # 初始化top_features变量，确保在任何情况下都有定义
        top_features = []
        
        # 评估特征重要性
        feature_importance = feature_selector.evaluate_feature_importance(train_data, test_data)
        
        if feature_importance is not None:
            # 获取前20个最重要的特征
            top_features = feature_selector.get_top_features(n_features=20)
            
            # 使用筛选后的特征
            train_data = train_data[top_features]
            test_data = test_data[top_features]
            
            print(f'\n当前最重要的{len(top_features)}个风险指标：')
            for i, feature in enumerate(top_features, 1):
                print(f'{i}. {feature}')
            
            # 检测异常
            print('\n开始检测重要指标的异常...')
            anomaly_results = feature_selector.detect_anomalies(test_data)
            
            if anomaly_results is not None:
                # 统计每个特征的异常数量
                anomaly_counts = {}
                for feature in top_features:
                    if feature in anomaly_results:
                        result = anomaly_results[feature]
                        anomaly_count = result['is_anomaly'].sum()
                        if anomaly_count > 0:
                            anomaly_counts[feature] = anomaly_count
                
                if anomaly_counts:
                    print('\n检测到以下重要指标存在异常：')
                    for feature, count in sorted(anomaly_counts.items(), key=lambda x: x[1], reverse=True):
                        print(f'{feature}: {count}个异常点')
                else:
                    print('\n未检测到重要指标的异常')
            else:
                print('异常检测失败')
        else:
            print('特征重要性评估失败')
            # 如果特征重要性评估失败，使用所有特征
            top_features = train_data.columns.tolist()
            if '日期' in top_features:
                top_features.remove('日期')
            print(f'使用所有{len(top_features)}个特征继续分析')
            
    except Exception as e:
        print(f'特征筛选阶段出错：{str(e)}')
        # 如果出错，使用所有特征
        top_features = train_data.columns.tolist()
        if '日期' in top_features:
            top_features.remove('日期')
        print(f'使用所有{len(top_features)}个特征继续分析')

    # 步骤4：风险监测系统
    print('\n4. 风险监测系统阶段')
    try:
        # 导入风险监测系统
        from risk_monitoring import RiskMonitoringSystem, generate_monitoring_report
        
        # 创建风险监测系统实例
        risk_monitor = RiskMonitoringSystem()
        
        # 计算预警阈值
        print('\n计算预警阈值...')
        warning_thresholds = risk_monitor.calculate_warning_thresholds(train_data, top_features)
        
        # 检查风险预警
        print('\n检查风险预警情况...')
        warnings = risk_monitor.check_warning_indicators(test_data, warning_thresholds)
        if warnings:
            print('发现以下风险预警：')
            for warning in warnings:
                print(f'- {warning}')
        else:
            print('未发现显著风险预警')
        
        # 异常检测
        print('\n执行异常检测...')
        anomalies = risk_monitor.detect_anomalies(test_data)
        if anomalies:
            print('检测到以下指标异常：')
            for anomaly in anomalies:
                print(f'- {anomaly}')
        else:
            print('未检测到显著异常')
        
        # 生成风险监测报告
        print('\n生成风险监测报告...')
        report_path = generate_monitoring_report(
            data=test_data,
            warning_thresholds=warning_thresholds,
            top_features=top_features,
            warnings=warnings,
            anomalies=anomalies
        )
        
        print(f'\n风险监测报告已生成：{report_path}')
        print('\n风险监测完成')
        
    except Exception as e:
        print(f'风险监测阶段出错：{str(e)}')
        return

if __name__ == '__main__':
    main()