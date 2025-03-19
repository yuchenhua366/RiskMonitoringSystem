# 金融风险监测系统

## feature_engineering.py
1. create_time_series_features函数：
basic_cols = [
                '贷款余额', '存款余额', '票据承兑余额', '存放同业余额', '投资余额',
                '当日付款金额', '当日收款金额', '净资金流入', '流动资产', '流动负债'
            ]

按最终修改
2. create_volatility_features.函数：
basic_cols = [
                '贷款余额', '存款余额', '票据承兑余额', '存放同业余额', '投资余额',
                '当日付款金额', '当日收款金额', '净资金流入', '流动资产', '流动负债'
            ]

按最终修改
3.create_composite_risk_features函数：
需注意：
   if all(col in data.columns for col in ['流动资产', '流动负债']):
                basic_risk_df['流动比率'] = data['流动负债'] / data['流动资产']
                basic_risk_df['流动性缺口率'] = (data['流动资产'] - data['流动负债']) / data['流动负债']

4. create_global_time_series_features函数：
  # 基础指标列表
        basic_cols = ['当日付款金额', '当日收款金额', '净资金流入', 
                     '存款余额', '贷款余额', '存放同业余额', '投资余额']
        
        # 比率指标列表
        ratio_cols = ["流动比率", "备付比率", "基金收益率", "前5大客户存款占比"]
索引没有对齐的代码，需改进
建议每个函数用Merge操作，保证索引对齐，最后再使用Merge对齐一次

## feature_selection.py
1. isolation_forest只是在筛选完重要指标后使用，并未参与重要指标的筛选

2. evaluate_feature_importance函数：
 importance_scores = pd.DataFrame({
                'feature': features,
                'variance_score': weighted_variance,
                'correlation_score': avg_correlation,
                'pca_score': pca_importance
            })


    # 1. 方差分析（带时间权重）- 使用标准化后的数据
            weighted_variance = np.average(X**2, weights=time_weights, axis=0) - \
                              np.average(X, weights=time_weights, axis=0)**2
            
            # 2. 相关性分析（带时间权重）- 使用标准化后的数据
            correlation_matrix = X_scaled_df.corr().abs()
            avg_correlation = correlation_matrix.mean()
            
            # 3. PCA分析（带时间权重）
            weighted_X = X * time_weights.reshape(-1, 1)
            self.pca.fit(weighted_X)
            component_weights = np.abs(self.pca.components_)
            # 改进点1：增加主成分解释方差权重
            explained_weights = self.pca.explained_variance_ratio_
            weighted_components = component_weights * explained_weights.reshape(-1, 1)
            pca_importance = np.sum(weighted_components, axis=0)  # 改进点2：加权求和

方差分析标准化的数据，是否合适？

## main.py
anomaly_results = feature_selector.detect_anomalies(test_data)
只是test_data而非全部数据