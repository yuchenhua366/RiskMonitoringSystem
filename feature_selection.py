import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from scipy import stats

class FeatureSelection:
    def __init__(self):
        self.selected_features = None
        self.scaler = StandardScaler()
        self.variance_selector = VarianceThreshold(threshold=0)
        self.pca = PCA(n_components=0.95)  # 保留95%的方差
        self.isolation_forest = IsolationForest(contamination=0.02, random_state=42)
        self.feature_importance = {}
        
    def evaluate_feature_importance(self, train_data, test_data):
        """使用多维度方法评估特征重要性，并考虑时间衰减因子"""
        try:
            # 合并数据集
            data = pd.concat([train_data, test_data], axis=0)
            
            # 计算时间衰减权重
            if '日期' in data.columns:
                data['日期'] = pd.to_datetime(data['日期'])
                latest_date = data['日期'].max()
                days_diff = (latest_date - data['日期']).dt.days
                time_weights = np.exp(-days_diff / 30)  # 30天的半衰期
            else:
                # 如果没有日期列，使用行索引作为时间参考
                time_weights = np.exp(-np.arange(len(data))[::-1] / 30)
            
            # 首先识别比率类指标和数值类指标
            ratio_features = []
            numeric_features = []
            for col in data.columns:
                if col == '日期':
                    continue
                if any(keyword in str(col).lower() for keyword in ['比', '率', '占比', '集中度']):
                    ratio_features.append(col)
                else:
                    numeric_features.append(col)
            
            # 分别对比率类指标和数值类指标进行标准化处理
            X_ratio = None
            X_numeric = None
            
            if ratio_features:
                ratio_data = data[ratio_features]
                ratio_scaler = StandardScaler()
                X_ratio = ratio_scaler.fit_transform(ratio_data)
                X_ratio_df = pd.DataFrame(X_ratio, columns=ratio_features, index=data.index)
            
            if numeric_features:
                numeric_data = data[numeric_features]
                numeric_scaler = StandardScaler()
                X_numeric = numeric_scaler.fit_transform(numeric_data)
                X_numeric_df = pd.DataFrame(X_numeric, columns=numeric_features, index=data.index)
            
            # 合并标准化后的数据
            if X_ratio is not None and X_numeric is not None:
                X = np.hstack((X_ratio, X_numeric))
                X_scaled_df = pd.concat([X_ratio_df, X_numeric_df], axis=1)
            elif X_ratio is not None:
                X = X_ratio
                X_scaled_df = X_ratio_df
            elif X_numeric is not None:
                X = X_numeric
                X_scaled_df = X_numeric_df
            else:
                raise ValueError('没有有效的特征可供分析')
            
            # 保存原始数据用于方差分析
            if X_ratio is not None and X_numeric is not None:
                X_original = np.hstack((ratio_data.values, numeric_data.values))
            elif X_ratio is not None:
                X_original = ratio_data.values
            elif X_numeric is not None:
                X_original = numeric_data.values
            
            # 1. 方差分析（带时间权重）- 使用未标准化的数据
            weighted_variance = np.average(X_original**2, weights=time_weights, axis=0) - \
                              np.average(X_original, weights=time_weights, axis=0)**2
            
            # 2. 相关性分析（带时间权重）- 使用标准化后的数据
            correlation_matrix = X_scaled_df.corr().abs()
            avg_correlation = correlation_matrix.mean()
            correlation_std = correlation_matrix.std()
            avg_correlation = avg_correlation + 0.5 * correlation_std
            
            # 3. PCA分析（带时间权重）- 使用标准化后的数据
            weighted_X = X_scaled_df * time_weights.reshape(-1, 1)
            self.pca.fit(weighted_X)
            component_weights = np.abs(self.pca.components_)
            explained_weights = self.pca.explained_variance_ratio_
            weighted_components = component_weights * explained_weights.reshape(-1, 1)
            pca_importance = np.sum(weighted_components, axis=0)
            
            # 4. 孤立森林分析 - 评估特征的独特性和异常检测能力
            isolation_importance = np.zeros(X_scaled_df.shape[1])
            recent_data = data.iloc[-7:]  # 使用原始数据的最近7天
            
            for i, feature in enumerate(X_scaled_df.columns):
                # 使用原始数据进行孤立森林分析
                feature_data = data[feature].values.reshape(-1, 1)
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                iso_forest.fit(feature_data)
                
                # 计算最近数据的异常分数
                recent_feature_data = recent_data[feature].values.reshape(-1, 1)
                anomaly_scores = iso_forest.score_samples(recent_feature_data)
                
                # 计算特征重要性：异常检测能力 + 时间权重
                recent_weights = time_weights[-7:]
                weighted_anomaly_score = np.average(np.abs(anomaly_scores), weights=recent_weights)
                isolation_importance[i] = weighted_anomaly_score
            
            # 综合评分
            features = X_scaled_df.columns
            # 确保is_ratio列正确设置
            is_ratio_values = [feature in ratio_features for feature in features]
            
            importance_scores = pd.DataFrame({
                'feature': features,
                'variance_score': weighted_variance,
                'correlation_score': avg_correlation,
                'pca_score': pca_importance,
                'isolation_score': isolation_importance,
                'is_ratio': is_ratio_values
            })
            
            # 确保is_ratio列存在且类型正确
            importance_scores['is_ratio'] = importance_scores['is_ratio'].astype(bool)
            
            # 归一化各个得分
            for col in ['variance_score', 'correlation_score', 'pca_score', 'isolation_score']:
                importance_scores[col] = (importance_scores[col] - importance_scores[col].min()) / \
                                      (importance_scores[col].max() - importance_scores[col].min() + 1e-10)
            
            # 更新计算综合得分的权重
            def calculate_importance(row):
                """计算特征重要性得分。
                
                比率类指标: 方差(15%), 相关性(25%), PCA(30%), 孤立森林(30%)
                数值类指标: 方差(20%), 相关性(20%), PCA(30%), 孤立森林(30%)
                """
                if row['is_ratio']:
                    return (
                        row['variance_score'] * 0.15 +
                        row['correlation_score'] * 0.25 +
                        row['pca_score'] * 0.30 +
                        row['isolation_score'] * 0.30
                    )
                else:
                    return (
                        row['variance_score'] * 0.20 +
                        row['correlation_score'] * 0.20 +
                        row['pca_score'] * 0.30 +
                        row['isolation_score'] * 0.30
                    )
            
            importance_scores['importance'] = importance_scores.apply(calculate_importance, axis=1)
            
            # 对比率类指标和数值类指标分别进行归一化，确保两类指标的重要性分数在相同的尺度上
            ratio_scores = importance_scores[importance_scores['is_ratio']]
            numeric_scores = importance_scores[~importance_scores['is_ratio']]
            
            if not ratio_scores.empty:
                ratio_min = ratio_scores['importance'].min()
                ratio_max = ratio_scores['importance'].max()
                if ratio_max > ratio_min:  # 避免除以零
                    importance_scores.loc[importance_scores['is_ratio'], 'importance'] = \
                        (ratio_scores['importance'] - ratio_min) / (ratio_max - ratio_min)
            
            if not numeric_scores.empty:
                numeric_min = numeric_scores['importance'].min()
                numeric_max = numeric_scores['importance'].max()
                if numeric_max > numeric_min:  # 避免除以零
                    importance_scores.loc[~importance_scores['is_ratio'], 'importance'] = \
                        (numeric_scores['importance'] - numeric_min) / (numeric_max - numeric_min)
            
            # 确保比率类指标和数值类指标的重要性分数分布均衡
            # 分别对比率类和数值类指标进行排序
            ratio_scores = importance_scores[importance_scores['is_ratio']].sort_values('importance', ascending=False)
            numeric_scores = importance_scores[~importance_scores['is_ratio']].sort_values('importance', ascending=False)
            
            # 交替合并两类指标，确保最终结果中两类指标都有代表
            merged_scores = []
            max_len = max(len(ratio_scores), len(numeric_scores))
            
            for i in range(max_len):
                if i < len(ratio_scores):
                    merged_scores.append(ratio_scores.iloc[i])
                if i < len(numeric_scores):
                    merged_scores.append(numeric_scores.iloc[i])
            
            self.feature_importance = pd.DataFrame(merged_scores)
            return self.feature_importance
            
        except Exception as e:
            print(f'评估特征重要性时出错：{str(e)}')
            return None
    
    def detect_anomalies(self, data, window_size=30, time_window=None):
        """检测特征的异常变化"""
        try:
            anomaly_results = {}
            
            # 根据时间窗口选择数据
            if time_window == 'last_day':
                detection_data = data.iloc[-1:].copy()
            elif time_window == 'last_week':
                detection_data = data.iloc[-7:].copy()
            else:
                detection_data = data.copy()
            
            for feature in detection_data.columns:
                feature_data = detection_data[feature].values.reshape(-1, 1)
                
                # 使用完整数据计算统计量，以获得更可靠的基线
                full_feature_data = data[feature].values.reshape(-1, 1)
                
                # 1. 移动窗口统计
                rolling_mean = pd.Series(full_feature_data.ravel()).rolling(window=window_size).mean()
                rolling_std = pd.Series(full_feature_data.ravel()).rolling(window=window_size).std()
                
                # 获取检测数据对应的统计量
                if time_window == 'last_day':
                    rolling_mean = rolling_mean.iloc[-1:]
                    rolling_std = rolling_std.iloc[-1:]
                elif time_window == 'last_week':
                    rolling_mean = rolling_mean.iloc[-7:]
                    rolling_std = rolling_std.iloc[-7:]
                
                # 2. Z-score异常检测
                # 使用完整数据计算均值和标准差
                feature_mean = np.mean(full_feature_data)
                feature_std = np.std(full_feature_data)
                z_scores = np.abs((feature_data - feature_mean) / (feature_std + 1e-10))
                
                # 3. 隔离森林异常检测
                # 使用完整数据训练隔离森林，调整contamination与特征选择保持一致
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                iso_forest.fit(full_feature_data)
                isolation_scores = iso_forest.score_samples(feature_data)
                
                # 综合评估异常
                anomaly_scores = pd.DataFrame({
                    'timestamp': detection_data.index,
                    'value': feature_data.ravel(),
                    'rolling_mean': rolling_mean,
                    'rolling_std': rolling_std,
                    'z_score': z_scores.ravel(),
                    'isolation_score': isolation_scores
                })
                
                # 标记异常
                anomaly_scores['is_anomaly'] = (
                    (z_scores > 3).ravel() |  # Z-score > 3
                    (isolation_scores < -0.3)  # 调整阈值，与contamination=0.1更匹配
                )
                
                anomaly_results[feature] = anomaly_scores
            
            return anomaly_results
            
        except Exception as e:
            print(f'检测异常时出错：{str(e)}')
            return None
    
    def select_features(self, importance_threshold=0.5):
        """根据特征重要性阈值筛选特征"""
        if self.feature_importance is None:
            raise ValueError('请先运行特征重要性评估')
        
        selected_features = self.feature_importance[
            self.feature_importance['importance'] > importance_threshold
        ]['feature'].tolist()
        
        self.selected_features = selected_features
        return selected_features
    
    def extract_base_feature_name(self, feature_name):
        """提取指标的基本名称，去除统计量后缀
        例如：将"存贷比_Q3中位数"转换为"存贷比"
        """
        import re
        
        # 处理时间窗口和统计量后缀
        patterns = [
            r'_[MQ]\d+.*$',  # 匹配如 _M6均值, _Q3中位数 等
            r'_\d+日.*$',  # 匹配如 _14日均值, _30日变化率 等
            r'_\d+月.*$',  # 匹配如 _3月均值 等
            r'_均值$', r'_中位数$', r'_标准差$', r'_方差$', r'_偏度$', r'_峰度$',
            r'_波动率$', r'_变化率$', r'_变化$', r'_增长率$', r'_环比$', r'_同比$',
            r'_最大值$', r'_最小值$', r'_极差$', r'_分位数$'
        ]
        
        base_name = feature_name
        for pattern in patterns:
            base_name = re.sub(pattern, '', base_name)
        
        # 处理特殊情况：前N大客户相关指标
        if re.match(r'^前\d+大客户', base_name):
            # 将"前1大客户存款占比"和"前2大客户存款占比"视为同一类
            base_name = re.sub(r'^前\d+大客户', '前N大客户', base_name)
        
        # 处理特殊情况：到期期限相关指标
        if any(term in base_name for term in ['到期', '期限']):
            # 将不同期限的指标归为同一类
            base_name = re.sub(r'\d+-\d+[天月年]|\d+[天月年]以上?', '期限', base_name)
        
        return base_name

    def get_top_features(self, n_features=20):
        """获取最重要的前N个特征，确保比率类和数值类指标都有代表，并避免选择重复意义的指标"""
        if self.feature_importance is None:
            raise ValueError('请先运行特征重要性评估')
        
        # 分别获取比率类指标和数值类指标
        ratio_features = self.feature_importance[self.feature_importance['is_ratio']].sort_values('importance', ascending=False)
        numeric_features = self.feature_importance[~self.feature_importance['is_ratio']].sort_values('importance', ascending=False)
        
        # 去除重复意义的指标，只保留每类指标中重要性最高的一个
        def extract_unique_features(features_df):
            unique_features = []
            processed_base_names = set()
            
            for _, row in features_df.iterrows():
                feature = row['feature']
                base_name = self.extract_base_feature_name(feature)
                if base_name not in processed_base_names:
                    unique_features.append(row)
                    processed_base_names.add(base_name)
            
            return pd.DataFrame(unique_features)
        
        # 对比率类和数值类指标分别去重
        unique_ratio_features = extract_unique_features(ratio_features)
        unique_numeric_features = extract_unique_features(numeric_features)
        
        # 计算每类指标应选取的数量
        ratio_count = min(len(unique_ratio_features), n_features // 2 + n_features % 2)  # 比率类指标数量（向上取整）
        numeric_count = min(len(unique_numeric_features), n_features // 2)  # 数值类指标数量
        
        # 如果某一类指标不足，则从另一类补充
        if ratio_count < n_features // 2 + n_features % 2:
            numeric_count = min(len(unique_numeric_features), n_features - ratio_count)
        if numeric_count < n_features // 2:
            ratio_count = min(len(unique_ratio_features), n_features - numeric_count)
        
        # 获取两类指标的前N个
        top_ratio_features = unique_ratio_features.head(ratio_count)['feature'].tolist()
        top_numeric_features = unique_numeric_features.head(numeric_count)['feature'].tolist()
        
        # 合并两类指标
        top_features = top_ratio_features + top_numeric_features
        
        # 根据原始重要性分数排序
        feature_importance_dict = dict(zip(self.feature_importance['feature'], self.feature_importance['importance']))
        top_features.sort(key=lambda x: feature_importance_dict.get(x, 0), reverse=True)
        
        return top_features
    
