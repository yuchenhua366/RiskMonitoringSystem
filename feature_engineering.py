import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold

class FeatureEngineering:
    def __init__(self):
        self.feature_importance = {}
        self.ensemble_importance = None
        
    def process(self, data):
        """特征工程主函数"""
        try:
            # 创建数据副本以避免修改原始数据
            processed_data = data.copy()
            
            # 确保日期列存在并且格式正确
            if '日期' in processed_data.columns:
                try:
                    # 尝试将日期列转换为datetime格式
                    processed_data['日期'] = pd.to_datetime(processed_data['日期'], format='%Y-%m-%d')
                    # 将日期列设置为索引
                    processed_data.set_index('日期', inplace=True)
                    # 确保数据按时间排序
                    processed_data = processed_data.sort_index()
                except Exception as e:
                    print(f'日期转换出错：{str(e)}')
                    # 如果转换失败，尝试其他常见格式
                    try:
                        processed_data['日期'] = pd.to_datetime(processed_data['日期'])
                        processed_data.set_index('日期', inplace=True)
                        processed_data = processed_data.sort_index()
                    except:
                        print('无法解析日期格式，请检查日期列格式是否正确')
                        return data
            
            # 创建一个列表来存储所有特征DataFrame
            feature_dfs = [processed_data]
            
            # 创建时序特征
            time_series_features = self.create_time_series_features(processed_data)
            if time_series_features is not None:
                feature_dfs.append(time_series_features)
            
            # 创建全局时序特征
            global_features = self.create_global_time_series_features(processed_data)
            if global_features is not None:
                feature_dfs.append(global_features)
            
            # 创建波动率指标
            volatility_features = self.create_volatility_features(processed_data)
            if volatility_features is not None:
                feature_dfs.append(volatility_features)
            
            # 创建复合风险指标
            composite_features = self.create_composite_risk_features(processed_data)
            if composite_features is not None:
                feature_dfs.append(composite_features)
            
            # 检查是否有空的DataFrame，如果有则移除
            non_empty_dfs = [df for df in feature_dfs if df is not None and not df.empty]
            if non_empty_dfs:
                # 确保所有DataFrame的索引一致
                for i in range(len(non_empty_dfs)):
                    if not non_empty_dfs[i].index.equals(processed_data.index):
                        non_empty_dfs[i] = non_empty_dfs[i].reindex(processed_data.index)
                # 一次性合并所有特征DataFrame
                processed_data = pd.concat(non_empty_dfs, axis=1, join='inner')
                # 处理可能的重复列
                processed_data = processed_data.loc[:,~processed_data.columns.duplicated()]
            
            # 如果日期是索引，先重置为普通列
            if isinstance(processed_data.index, pd.DatetimeIndex):
                processed_data = processed_data.reset_index()

            # 删除日期列
            if '日期' in processed_data.columns:
                processed_data = processed_data.drop('日期', axis=1)

            # 确保索引是从1开始的数字
            processed_data.index = range(1, len(processed_data) + 1)
            
            # 删除重复列和空值列
            processed_data_t = processed_data.T  # 列转行
            processed_data_t = processed_data_t.drop_duplicates()  # 删除重复的行（原重复列）
            processed_data = processed_data_t.T
            processed_data.dropna(axis=1, how='all', inplace=True)
            processed_data = processed_data.replace(0, pd.NA).dropna(axis=1, how='all')
            
            # 删除不需要的时间特征列
            if 'quarter' in processed_data.columns:
                processed_data.drop(columns=["quarter", "month", "day_of_week"], inplace=True)
            
            # 保存处理后的数据
            processed_data.to_csv('processed_data.csv')
            
            # 划分训练集和测试集（使用最新的7天数据作为测试集）
            train_data = processed_data[:-7]
            test_data = processed_data[-7:]
            
            # 处理缺失值
            train_data, test_data = self.MissingValueThreshold(train_data, test_data)
            
            # 保存训练集和测试集
            train_data.to_csv('train_data.csv')
            test_data.to_csv('test_data.csv')
            
            return train_data, test_data
            
        except Exception as e:
            print(f'特征工程处理过程中出错：{str(e)}')
            return None, None  # 发生错误时返回None
            
    def MissingValueThreshold(self,X_train_temp, X_test_temp, threshold = 0.5, fn = 0):
        """
        根据比例删除缺失值比例较高的特征
        同时将其他缺失值统一填补为fn的值
        
        :param X_train_temp: 训练集特征
        :param X_test_temp: 测试集特征
        :param threshold: 缺失值比例阈值
        :param fn: 其他缺失值填补数值
        
        :return：剔除指定特征后的X_train_temp和X_test_temp
        """
        for col in X_train_temp:
            if X_train_temp[col].isnull().sum() / X_train_temp.shape[0] >= threshold:
                del X_train_temp[col]
                del X_test_temp[col]
            else:
                X_train_temp.loc[:, col] = X_train_temp.loc[:, col].fillna(fn)
                X_test_temp.loc[:, col] = X_test_temp.loc[:, col].fillna(fn)
        return X_train_temp, X_test_temp
    
    def create_time_series_features(self, data):
        """创建时序特征"""
        # 创建数据副本以避免修改原始数据
        data = data.copy()
        
        # 确保输入数据有正确的日期索引
        if not isinstance(data.index, pd.DatetimeIndex):
            if '日期' in data.columns:
                try:
                    data['日期'] = pd.to_datetime(data['日期'])
                    data = data.set_index('日期')
                except Exception as e:
                    print(f'日期转换出错：{str(e)}')
                    return data
            else:
                print('无法找到日期列，无法创建时序特征')
                return data
        
        try:
            # 创建一个列表来存储所有特征DataFrame
            feature_dfs = [data.copy()]
            
            # 扩展基础指标列表，包含更多的资金流动相关指标
            basic_cols = [
                '贷款余额', '存款余额', '票据承兑余额', '存放同业余额', '投资余额',
                '当日付款金额', '当日收款金额', '净资金流入', '流动资产', '流动负债'
            ]
            
            # 添加时间特征
            time_df = pd.DataFrame(index=data.index)
            time_df['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
            time_df['quarter'] = data.index.quarter
            time_df['month'] = data.index.month
            time_df['day_of_week'] = data.index.dayofweek + 1
            feature_dfs.append(time_df)
            
            # 计算各个指标的变化率
            for col in basic_cols:
                if col in data.columns:
                    temp_df = pd.DataFrame(index=data.index)
                    for window in [3, 7, 14, 30]:
                        # 使用安全的方式计算变化率，避免NaN传播
                        temp_df[f'{col}_{window}日变化率'] = data[col].pct_change(window)
                    feature_dfs.append(temp_df)
            
            # 计算比率指标
            ratio_df = pd.DataFrame(index=data.index)
            
            # 存贷比
            if all(col in data.columns for col in ['贷款余额', '存款余额']):
                ratio_df['存贷比'] = data['贷款余额'] / data['存款余额']
            
            # 投资存款比
            if all(col in data.columns for col in ['投资余额', '存款余额']):
                ratio_df['投资存款比'] = data['投资余额'] / data['存款余额']
            
            # 同业存款比
            if all(col in data.columns for col in ['存放同业余额', '存款余额']):
                ratio_df['同业存款比'] = data['存放同业余额'] / data['存款余额']
                
            # 净流入比率
            if all(col in data.columns for col in ['净资金流入', '存款余额']):
                ratio_df['净流入存款比'] = data['净资金流入'] / data['存款余额']
                
            # 付款收款比
            if all(col in data.columns for col in ['当日付款金额', '当日收款金额']):
                ratio_df['付款收款比'] = data['当日付款金额'] / data['当日收款金额']
                
            # 添加比率指标的变化率
            for col in ratio_df.columns:
                for window in [3, 7, 14, 30]:
                    ratio_df[f'{col}_{window}日变化率'] = ratio_df[col].pct_change(window)
                    
            feature_dfs.append(ratio_df)
            
            # 按工作日/周末分组统计特征
            workday_weekend_df = pd.DataFrame(index=data.index)
            for col in basic_cols:
                if col in data.columns:
                    # 创建工作日/周末标记
                    is_weekend = data.index.dayofweek.isin([5, 6])
                    
                    # 计算滚动窗口内工作日和周末的统计量
                    for window in [7, 14, 30]:
                        # 创建临时Series用于存储工作日和周末的值
                        workday_series = pd.Series(index=data.index, dtype=float)
                        weekend_series = pd.Series(index=data.index, dtype=float)
                        
                        # 分别计算工作日和周末的值，并确保索引对齐
                        workday_data = data[col][~is_weekend]
                        weekend_data = data[col][is_weekend]
                        
                        # 使用rolling而不是expanding来计算移动平均，并确保索引对齐
                        workday_values = workday_data.rolling(window=window, min_periods=1).mean()
                        weekend_values = weekend_data.rolling(window=window, min_periods=1).mean()
                        
                        # 将计算结果分配到对应的索引位置，使用reindex确保索引对齐
                        workday_series.loc[workday_values.index] = workday_values
                        weekend_series.loc[weekend_values.index] = weekend_values
                        
                        # 使用前向填充和后向填充处理缺失值
                        workday_series = workday_series.ffill().bfill()
                        weekend_series = weekend_series.ffill().bfill()
                        
                        # 将结果添加到特征DataFrame
                        workday_weekend_df[f'{col}_工作日均值_{window}日'] = workday_series
                        workday_weekend_df[f'{col}_周末均值_{window}日'] = weekend_series
            
            feature_dfs.append(workday_weekend_df)
            
            # 计算同比和环比变化
            seasonal_df = pd.DataFrame(index=data.index)
            for col in basic_cols:
                if col in data.columns:
                    # 计算环比变化（与上月相比）
                    seasonal_df[f'{col}_环比变化'] = data[col].pct_change()
                    # 计算同比变化（与去年同月相比，如果有足够的数据）
                    if len(data) >= 365:  # 确保有足够的历史数据
                        seasonal_df[f'{col}_同比变化'] = data[col].pct_change(365)
            
            feature_dfs.append(seasonal_df)
            
            # 使用merge操作确保所有DataFrame的索引对齐
            for i in range(1, len(feature_dfs)):
                if not feature_dfs[i].empty:
                    feature_dfs[i] = feature_dfs[i].reset_index()
                    feature_dfs[i] = pd.merge(data.reset_index(), feature_dfs[i], on='日期', how='left')
                    feature_dfs[i] = feature_dfs[i].set_index('日期')
            
            # 使用pd.concat一次性合并所有特征，确保索引对齐
            # 检查是否有空的DataFrame，如果有则移除
            non_empty_dfs = [df for df in feature_dfs if not df.empty]
            if non_empty_dfs:
                result = pd.concat(non_empty_dfs, axis=1)
                # 使用均值填充处理缺失值
                result = result.fillna(result.mean())  # 先用均值填充
                result = result.fillna(0)  # 将剩余的NaN值填充为0
                return result
            else:
                return data  # 如果没有有效的特征DataFrame，返回原始数据
        except Exception as e:
            print(f'创建时序特征时出错：{str(e)}')
            return data  # 发生错误时返回原始数据
    
    def create_volatility_features(self, data):
        """创建波动率指标"""
        try:
            # 创建数据副本以避免修改原始数据
            data = data.copy()
            
            # 确保输入数据有正确的日期索引
            if not isinstance(data.index, pd.DatetimeIndex):
                if '日期' in data.columns:
                    try:
                        data['日期'] = pd.to_datetime(data['日期'])
                        data = data.set_index('日期')
                    except Exception as e:
                        print(f'日期转换出错：{str(e)}')
                        return None
                else:
                    print('无法找到日期列，无法创建波动率指标')
                    return None
            
            # 创建一个空的DataFrame列表来存储所有特征
            feature_dfs = []
            
            # 扩展基础指标列表，包含更多的资金流动相关指标
            basic_cols = [
                '贷款余额', '存款余额', '票据承兑余额', '存放同业余额', '投资余额',
                '当日付款金额', '当日收款金额', '净资金流入', '流动资产', '流动负债'
            ]
            
            # 计算各个指标的波动率特征
            for col in basic_cols:
                if col in data.columns:
                    temp_df = pd.DataFrame(index=data.index)
                    for window in [3, 7, 14, 30]:
                        # 计算滚动特征
                        rolling = data[col].rolling(window=window, min_periods=1)
                        temp_df[f'{col}_{window}日均值'] = rolling.mean()
                        temp_df[f'{col}_{window}日标准差'] = rolling.std()
                        # 处理可能的除零问题
                        mean_values = rolling.mean()
                        std_values = rolling.std()
                        temp_df[f'{col}_{window}日波动率'] = np.where(
                            mean_values != 0,
                            std_values / mean_values,
                            0
                        )
                        temp_df[f'{col}_{window}日偏度'] = rolling.skew()
                        temp_df[f'{col}_{window}日峰度'] = rolling.kurt()
                    feature_dfs.append(temp_df)
            
            # 计算比率指标
            ratio_df = pd.DataFrame(index=data.index)
            
            # 存贷比
            if all(col in data.columns for col in ['贷款余额', '存款余额']):
                ratio_df['存贷比'] = data['贷款余额'] / data['存款余额']
            
            # 投资存款比
            if all(col in data.columns for col in ['投资余额', '存款余额']):
                ratio_df['投资存款比'] = data['投资余额'] / data['存款余额']
            
            # 同业存款比
            if all(col in data.columns for col in ['存放同业余额', '存款余额']):
                ratio_df['同业存款比'] = data['存放同业余额'] / data['存款余额']
            
            # 净流入比率
            if all(col in data.columns for col in ['净资金流入', '存款余额']):
                ratio_df['净流入存款比'] = data['净资金流入'] / data['存款余额']
            
            # 付款收款比
            if all(col in data.columns for col in ['当日付款金额', '当日收款金额']):
                ratio_df['付款收款比'] = data['当日付款金额'] / data['当日收款金额']
            
            # 计算比率指标的波动率特征
            if not ratio_df.empty:
                for col in ratio_df.columns:
                    temp_df = pd.DataFrame(index=data.index)
                    for window in [3, 7, 14, 30]:
                        rolling = ratio_df[col].rolling(window=window, min_periods=1)
                        temp_df[f'{col}_{window}日均值'] = rolling.mean()
                        temp_df[f'{col}_{window}日标准差'] = rolling.std()
                        mean_values = rolling.mean()
                        std_values = rolling.std()
                        temp_df[f'{col}_{window}日波动率'] = np.where(
                            mean_values != 0,
                            std_values / mean_values,
                            0
                        )
                        temp_df[f'{col}_{window}日偏度'] = rolling.skew()
                        temp_df[f'{col}_{window}日峰度'] = rolling.kurt()
                    feature_dfs.append(temp_df)
            
            # 使用merge操作确保所有DataFrame的索引对齐
            for i in range(len(feature_dfs)):
                if not feature_dfs[i].empty:
                    feature_dfs[i] = feature_dfs[i].reset_index()
                    feature_dfs[i] = pd.merge(data.reset_index(), feature_dfs[i], on='日期', how='left')
                    feature_dfs[i] = feature_dfs[i].set_index('日期')
            
            # 检查是否有空的DataFrame，如果有则移除
            non_empty_dfs = [df for df in feature_dfs if df is not None and not df.empty]
            if non_empty_dfs:
                result = pd.concat(non_empty_dfs, axis=1)
                # 使用均值填充处理缺失值
                result = result.fillna(result.mean())  # 先用均值填充
                result = result.fillna(0)  # 将剩余的NaN值填充为0
                return result
            else:
                return None
                
        except Exception as e:
            print(f'创建波动率指标时出错：{str(e)}')
            return None
    
    def create_composite_risk_features(self, data):
        """创建复合风险指标"""
        try:
            # 创建数据副本以避免修改原始数据
            data = data.copy()
            
            # 确保输入数据有正确的日期索引
            if not isinstance(data.index, pd.DatetimeIndex):
                if '日期' in data.columns:
                    try:
                        data['日期'] = pd.to_datetime(data['日期'])
                        data = data.set_index('日期')
                    except Exception as e:
                        print(f'日期转换出错：{str(e)}')
                        return data
                else:
                    print('无法找到日期列，无法创建复合风险指标')
                    return data
            
            # 创建一个列表来存储所有特征DataFrame
            feature_dfs = [data.copy()]
            
            # 创建一个临时DataFrame来存储基础风险指标
            basic_risk_df = pd.DataFrame(index=data.index)
            
            # 计算存贷比及其变化趋势
            if '贷款余额' in data.columns and '存款余额' in data.columns:
                basic_risk_df['存贷比'] = data['贷款余额'] / data['存款余额']
                basic_risk_df['存贷比变化趋势'] = basic_risk_df['存贷比'].diff()
            
            # 计算票据集中度
            if '票据承兑余额' in data.columns and '存放同业余额' in data.columns:
                basic_risk_df['票据集中度'] = data['票据承兑余额'] / data['存放同业余额']
            
            # 计算投资集中度及其变化趋势
            if '投资余额' in data.columns and '存款余额' in data.columns:
                basic_risk_df['投资集中度'] = data['投资余额'] / data['存款余额']
                basic_risk_df['投资集中度变化趋势'] = basic_risk_df['投资集中度'].diff()
            
            # 计算资金缺口覆盖率及其变化趋势
            if all(col in data.columns for col in ['流动资产', '流动负债']):
                basic_risk_df['资金缺口覆盖率'] = data['流动资产'] / data['流动负债']
                basic_risk_df['资金缺口覆盖率变化趋势'] = basic_risk_df['资金缺口覆盖率'].diff()
            
            # 添加流动性比率指标
            if all(col in data.columns for col in ['流动资产', '流动负债']):
                basic_risk_df['流动比率'] = data['流动负债'] / data['流动资产']
                basic_risk_df['流动性缺口率'] = (data['流动资产'] - data['流动负债']) / data['流动负债']
                
                # 计算流动比率的变化率
                for window in [3, 7, 14, 30]:
                    basic_risk_df[f'流动比率_{window}日变化率'] = basic_risk_df['流动比率'].pct_change(window)
                    basic_risk_df[f'流动比率_{window}日均值'] = basic_risk_df['流动比率'].rolling(window=window).mean()
            
            # 添加备付比率指标
            if all(col in data.columns for col in ['现金资产', '存款余额']):
                basic_risk_df['备付比率'] = data['现金资产'] / data['存款余额']
                
                # 计算备付比率的变化率
                for window in [3, 7, 14, 30]:
                    basic_risk_df[f'备付比率_{window}日变化率'] = basic_risk_df['备付比率'].pct_change(window)
            
            feature_dfs.append(basic_risk_df)
            
            # 计算错配率和资金缺口指标
            duration_pairs = [
                ('超短期', ['0-7天到期']),
                ('短期', ['7天-1个月到期', '1-3个月到期']),
                ('中期', ['3-6个月到期', '6-12个月到期']),
                ('长期', ['1年以上到期'])
            ]
            
            for duration, terms in duration_pairs:
                # 创建一个临时DataFrame来存储错配率和资金缺口指标
                mismatch_df = pd.DataFrame(index=data.index)
                
                # 计算存贷款错配率和资金缺口
                deposit_cols = [f'{term}存款' for term in terms]
                loan_cols = [f'{term}贷款' for term in terms]
                
                if all(col in data.columns for col in deposit_cols + loan_cols):
                    deposits = data[deposit_cols].sum(axis=1)
                    loans = data[loan_cols].sum(axis=1)
                    
                    # 计算资金缺口率
                    gap = loans - deposits
                    mismatch_df[f'{duration}资金缺口'] = gap
                    # 处理可能的除零问题
                    mismatch_df[f'{duration}资金缺口率'] = np.where(
                        deposits != 0,
                        gap / deposits,  # 已经是百分比格式，不需要乘以100
                        0
                    )
                    
                    # 计算错配率
                    mismatch_df[f'{duration}错配率'] = np.where(
                        deposits != 0,
                        loans / deposits,
                        0
                    )
                    
                    # 计算错配率和资金缺口率的变化
                    for window in [3, 7, 14, 30]:
                        mismatch_df[f'{duration}错配率_{window}日变化'] = mismatch_df[f'{duration}错配率'].diff(window)
                        mismatch_df[f'{duration}资金缺口率_{window}日变化'] = mismatch_df[f'{duration}资金缺口率'].diff(window)
                        
                        # 添加滚动统计特征
                        rolling = mismatch_df[f'{duration}资金缺口率'].rolling(window=window)
                        mismatch_df[f'{duration}资金缺口率_{window}日均值'] = rolling.mean()
                        mismatch_df[f'{duration}资金缺口率_{window}日标准差'] = rolling.std()
                        mismatch_df[f'{duration}资金缺口率_{window}日最大值'] = rolling.max()
                        mismatch_df[f'{duration}资金缺口率_{window}日最小值'] = rolling.min()
                        
                        # 添加资金缺口绝对值的滚动特征
                        abs_gap_rolling = np.abs(mismatch_df[f'{duration}资金缺口']).rolling(window=window)
                        mismatch_df[f'{duration}资金缺口绝对值_{window}日均值'] = abs_gap_rolling.mean()
                        mismatch_df[f'{duration}资金缺口绝对值_{window}日标准差'] = abs_gap_rolling.std()
                    
                    # 添加各期限段的具体资金缺口指标
                    for term in terms:
                        if f'{term}贷款' in data.columns and f'{term}存款' in data.columns:
                            term_deposits = data[f'{term}存款']
                            term_loans = data[f'{term}贷款']
                            term_gap = term_loans - term_deposits
                            
                            # 计算具体期限的资金缺口率
                            mismatch_df[f'{term}资金缺口率'] = np.where(
                                term_deposits != 0,
                                term_gap / term_deposits,  # 已经是百分比格式，不需要乘以100
                                0
                            )
                            
                            # 计算具体期限的错配率
                            mismatch_df[f'{term}错配率'] = np.where(
                                term_deposits != 0,
                                term_loans / term_deposits,
                                0
                            )
                            
                            # 添加具体期限的资金缺口绝对值
                            mismatch_df[f'{term}资金缺口绝对值'] = np.abs(term_gap)
                            
                            # 添加具体期限的资金缺口变化率
                            mismatch_df[f'{term}资金缺口变化率'] = term_gap.pct_change()
                            
                            # 添加具体期限的资金缺口占总资产比例
                            if '总资产' in data.columns:
                                mismatch_df[f'{term}资金缺口占总资产比例'] = np.where(
                                    data['总资产'] != 0,
                                    term_gap / data['总资产'],
                                    0
                                )
                            
                            # 添加具体期限的资金缺口占存款比例
                            if '存款余额' in data.columns:
                                mismatch_df[f'{term}资金缺口占存款比例'] = np.where(
                                    data['存款余额'] != 0,
                                    term_gap / data['存款余额'],
                                    0
                                )
                
                # 计算期限错配风险指标
                if duration != '超短期':
                    # 计算与上一期限的错配比
                    prev_duration = duration_pairs[duration_pairs.index((duration, terms))-1][0]
                    if f'{prev_duration}错配率' in mismatch_df.columns and f'{duration}错配率' in mismatch_df.columns:
                        mismatch_df[f'{prev_duration}_{duration}错配比'] = mismatch_df[f'{prev_duration}错配率'] / mismatch_df[f'{duration}错配率']
                        
                        # 计算错配比的变化趋势
                        mismatch_df[f'{prev_duration}_{duration}错配比变化'] = mismatch_df[f'{prev_duration}_{duration}错配比'].diff()
                
                # 计算流动性压力指标
                if duration == '超短期' or duration == '短期':
                    if f'{duration}资金缺口' in mismatch_df.columns and '存款余额' in data.columns:
                        mismatch_df[f'{duration}流动性压力指标'] = np.where(
                            data['存款余额'] != 0,
                            mismatch_df[f'{duration}资金缺口'] / data['存款余额'],
                            0
                        )
                        
                        # 计算流动性压力指标的变化趋势
                        mismatch_df[f'{duration}流动性压力指标变化'] = mismatch_df[f'{duration}流动性压力指标'].diff()
                
                feature_dfs.append(mismatch_df)
            
            # 使用merge操作确保所有DataFrame的索引对齐
            for i in range(1, len(feature_dfs)):
                if not feature_dfs[i].empty:
                    feature_dfs[i] = feature_dfs[i].reset_index()
                    feature_dfs[i] = pd.merge(data.reset_index(), feature_dfs[i], on='日期', how='left')
                    feature_dfs[i] = feature_dfs[i].set_index('日期')
            
            # 使用pd.concat一次性合并所有特征
            # 检查是否有空的DataFrame，如果有则移除
            non_empty_dfs = [df for df in feature_dfs if not df.empty]
            if non_empty_dfs:
                result = pd.concat(non_empty_dfs, axis=1)
                
                # 处理缺失值
                result = result.ffill().bfill()
                return result
            else:
                return None  # 如果没有有效的特征DataFrame，返回None
        except Exception as e:
            print(f'创建复合风险指标时出错：{str(e)}')
            return None  # 发生错误时返回None



    def create_global_time_series_features(self,df):
        """创建时序特征（全局函数版本）"""
        # 创建数据副本以避免修改原始数据
        df = df.copy()
        
        # 确保日期列格式正确
        if not isinstance(df.index, pd.DatetimeIndex):
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期')
            else:
                raise ValueError("DataFrame 中没有 '日期' 列，且索引也不是日期类型。请检查数据格式。")
        
        # 初始化一个空的特征列表
        feature_list = []
        
        # 添加时间特征
        time_features = pd.DataFrame(index=df.index)
        time_features['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        time_features['quarter'] = df.index.quarter
        time_features['month'] = df.index.month
        time_features['day_of_week'] = df.index.dayofweek + 1
        feature_list.append(time_features)
        
        # 基础指标列表
        basic_cols = ['当日付款金额', '当日收款金额', '净资金流入', 
                     '存款余额', '贷款余额', '存放同业余额', '投资余额']
        
        # 比率指标列表
        ratio_cols = ["流动比率", "备付比率", "基金收益率", "前5大客户存款占比"]
        
        # 数据类型转换：处理百分比格式的字符串
        for col in df.columns:
            if df[col].dtype == 'object':  # 检查是否为字符串类型
                # 尝试将百分比格式的字符串转换为数值类型
                df[col] = df[col].str.rstrip('%').replace('', '0').astype('float') / 100.0
            else:
                # 确保其他列是数值类型
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 动态计算比率指标
        if all(col in df.columns for col in ['贷款余额', '存款余额']):
            df['存贷比'] = df['贷款余额'] / df['存款余额']
            ratio_cols.append('存贷比')
        
        if all(col in df.columns for col in ['投资余额', '存款余额']):
            df['投资存款比'] = df['投资余额'] / df['存款余额']
            ratio_cols.append('投资存款比')
        
        if all(col in df.columns for col in ['存放同业余额', '存款余额']):
            df['同业存款比'] = df['存放同业余额'] / df['存款余额']
            ratio_cols.append('同业存款比')
        
        # 按工作日/周末分组统计特征
        for col in basic_cols + ratio_cols:
            if col in df.columns:
                # 创建工作日/周末标记
                is_weekend = df.index.dayofweek.isin([5, 6])
                
                # 计算滚动统计量
                for window in [7, 14, 30]:
                    # 工作日统计
                    workday_values = df.loc[~is_weekend, col].rolling(window=window, min_periods=1).mean()
                    workday_series = pd.Series(index=df.index, dtype=float)
                    workday_series.loc[~is_weekend] = workday_values
                    workday_series = workday_series.ffill().bfill()
                    feature_list.append(workday_series.rename(f'{col}_工作日均值_{window}日'))
                    
                    # 周末统计
                    weekend_values = df.loc[is_weekend, col].rolling(window=window, min_periods=1).mean()
                    weekend_series = pd.Series(index=df.index, dtype=float)
                    weekend_series.loc[is_weekend] = weekend_values
                    weekend_series = weekend_series.ffill().bfill()
                    feature_list.append(weekend_series.rename(f'{col}_周末均值_{window}日'))
        
        # 按季度分组统计特征
        for col in basic_cols + ratio_cols:
            if col in df.columns:
                for quarter in range(1, 5):
                    quarter_mask = df.index.quarter == quarter
                    quarter_data = df.loc[quarter_mask, col]
                    
                    # 创建季度特征
                    feature_list.append(quarter_data.expanding(min_periods=1).mean().rename(f'{col}_Q{quarter}均值'))
                    feature_list.append(quarter_data.expanding(min_periods=1).std().rename(f'{col}_Q{quarter}标准差'))
                    feature_list.append(quarter_data.expanding(min_periods=1).median().rename(f'{col}_Q{quarter}中位数'))
                    feature_list.append(quarter_data.expanding(min_periods=1).max().rename(f'{col}_Q{quarter}最大值'))
                    feature_list.append(quarter_data.expanding(min_periods=1).min().rename(f'{col}_Q{quarter}最小值'))
        
        # 按月份分组统计特征
        for col in basic_cols + ratio_cols:
            if col in df.columns:
                for month in range(1, 13):
                    month_mask = df.index.month == month
                    month_stats = df.loc[month_mask, col]
                    feature_list.append(month_stats.expanding(min_periods=1).mean().rename(f'{col}_M{month}均值'))
                    feature_list.append(month_stats.expanding(min_periods=1).std().rename(f'{col}_M{month}标准差'))
                    feature_list.append(month_stats.expanding(min_periods=1).median().rename(f'{col}_M{month}中位数'))
        
        # 计算同比和环比变化
        for col in basic_cols + ratio_cols:
            if col in df.columns:
                df[f'{col}_环比变化'] = df.groupby(df.index.year)[col].pct_change()
                df[f'{col}_同比变化'] = df.groupby(df.index.month)[col].pct_change(12)
                
                # 添加到特征列表
                feature_list.append(df[f'{col}_环比变化'].expanding(min_periods=1).mean().rename(f'{col}_环比变化_均值'))
                feature_list.append(df[f'{col}_环比变化'].expanding(min_periods=1).std().rename(f'{col}_环比变化_标准差'))
                feature_list.append(df[f'{col}_同比变化'].expanding(min_periods=1).mean().rename(f'{col}_同比变化_均值'))
                feature_list.append(df[f'{col}_同比变化'].expanding(min_periods=1).std().rename(f'{col}_同比变化_标准差'))
        
        # 计算指标间的相对变化
        if all(col in df.columns for col in ['存款余额', '贷款余额']):
            df['存贷变化比'] = df['存款余额'].pct_change() / df['贷款余额'].pct_change()
            feature_list.append(df['存贷变化比'].expanding(min_periods=1).mean().rename('存贷变化比_均值'))
            feature_list.append(df['存贷变化比'].expanding(min_periods=1).std().rename('存贷变化比_标准差'))
        
        if all(col in df.columns for col in ['投资余额', '存款余额']):
            df['投资存款变化比'] = df['投资余额'].pct_change() / df['存款余额'].pct_change()
            feature_list.append(df['投资存款变化比'].expanding(min_periods=1).mean().rename('投资存款变化比_均值'))
            feature_list.append(df['投资存款变化比'].expanding(min_periods=1).std().rename('投资存款变化比_标准差'))
        
        # 使用merge操作确保所有特征的索引对齐
        features_df = pd.DataFrame(index=df.index)
        for feature in feature_list:
            if not feature.empty:
                feature = feature.reset_index()
                feature = pd.merge(df.reset_index(), feature, on='日期', how='left')
                feature = feature.set_index('日期')
                features_df = pd.concat([features_df, feature], axis=1)
        
        # 使用前向填充和后向填充处理缺失值
        features_df = features_df.ffill().bfill()
        return features_df