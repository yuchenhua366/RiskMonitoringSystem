import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class RiskMonitoringSystem:
    def __init__(self):
        self.warning_thresholds = {}

    def calculate_warning_thresholds(self, data, features):
        """计算预警阈值，针对不同类型的指标采用不同的计算方法"""
        thresholds = {}
        for feature in features:
            if feature in data.columns:
                values = data[feature].dropna()
                if len(values) < 2:
                    continue
                
                # 判断指标类型
                is_ratio = any(keyword in feature.lower() for keyword in ['比', '率', '占比', '集中度'])
                is_change_rate = any(keyword in feature.lower() for keyword in ['变化率', '增长率', '变化', '波动'])
                
                # 计算基础统计量
                mean = values.mean()
                std = values.std()
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                p5 = values.quantile(0.05)
                p95 = values.quantile(0.95)
                
                # 计算EWMA统计量
                alpha = 0.1  # EWMA平滑系数
                ewma = values.ewm(alpha=alpha).mean().iloc[-1]
                ewma_std = values.ewm(alpha=alpha).std().iloc[-1]
                
                if is_ratio:
                    # 比率类指标：使用四分位距法，并考虑历史分位数
                    lower = max(q1 - 1.5 * iqr, p5)
                    upper = min(q3 + 1.5 * iqr, p95)
                elif is_change_rate:
                    # 变化率类指标：使用EWMA方法，考虑趋势
                    lower = ewma - 2.5 * ewma_std
                    upper = ewma + 2.5 * ewma_std
                else:
                    # 数值类指标：使用自适应标准差法
                    k = 2.5 if std/mean < 0.5 else 2.0  # 波动较小时使用更宽的区间
                    lower = mean - k * std
                    upper = mean + k * std
                
                # 确保上下限有合理的差异
                if upper <= lower:
                    margin = abs(mean * 0.05)
                    upper = mean + margin
                    lower = mean - margin
                
                thresholds[feature] = {
                    'lower': lower,
                    'upper': upper,
                    'mean': mean,
                    'std': std,
                    'type': 'ratio' if is_ratio else 'change_rate' if is_change_rate else 'numeric'
                }
        
        self.warning_thresholds = thresholds
        return thresholds

    def check_warning_indicators(self, data, warning_thresholds):
        """检查风险预警"""
        warnings = []
        for feature in warning_thresholds.keys():
            if feature in data.columns:
                current_value = data[feature].iloc[-1]
                thresholds = warning_thresholds[feature]
                # 对比率类指标进行百分比转换
                if thresholds['type'] == 'ratio':
                    display_value = current_value * 100
                    display_upper = thresholds['upper'] * 100
                    display_lower = thresholds['lower'] * 100
                    if current_value > thresholds['upper']:
                        warnings.append(f'{feature}超过上限阈值(当前值: {display_value:.2f}%, 上限: {display_upper:.2f}%)')
                    elif current_value < thresholds['lower']:
                        warnings.append(f'{feature}低于下限阈值(当前值: {display_value:.2f}%, 下限: {display_lower:.2f}%)')
                else:
                    # 数值类指标保持原始值
                    if current_value > thresholds['upper']:
                        warnings.append(f'{feature}超过上限阈值(当前值: {current_value:.2f}, 上限: {thresholds["upper"]:.2f})')
                    elif current_value < thresholds['lower']:
                        warnings.append(f'{feature}低于下限阈值(当前值: {current_value:.2f}, 下限: {thresholds["lower"]:.2f})')
        return warnings

    def detect_anomalies(self, data):
        """检测异常"""
        anomalies = []
        for column in data.columns:
            values = data[column].dropna()
            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                current_value = values.iloc[-1]
                
                # 使用更稳健的方法计算异常阈值
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # 如果标准差接近0，使用IQR方法判断异常
                if std < 1e-10:
                    if current_value < lower_bound or current_value > upper_bound:
                        anomalies.append(f'{column}出现异常波动')
                else:
                    # 使用z-score方法判断异常
                    z_score = abs((current_value - mean) / std)
                    if z_score > 3:
                        anomalies.append(f'{column}出现异常波动')
        return anomalies

def plot_indicator_trend(data, feature, warning_thresholds):
    """生成指标趋势图，显示近7天的指标值和预警阈值区间"""
    plt.figure(figsize=(10, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取最近7天数据
    recent_data = data[feature].iloc[-7:]
    dates = [f"Day{7-i}" for i in range(7)]
    
    if feature in warning_thresholds:
        thresholds = warning_thresholds[feature]
        upper = thresholds['upper']
        lower = thresholds['lower']
        
        # 创建图表背景
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')
        plt.grid(True, linestyle=':', color='#dddddd', alpha=0.7)
        
        # 绘制预警阈值线
        plt.axhline(y=upper, color='#e03131', linestyle='--', linewidth=1.2, label='预警上限')
        plt.axhline(y=lower, color='#e03131', linestyle='--', linewidth=1.2, label='预警下限')
        
        # 绘制指标值曲线
        plt.plot(dates, recent_data, color='#228be6', marker='o', 
                 linewidth=2.5, markersize=8, label='指标值', zorder=3)
        
        # 为每个点添加数值标签
        for i, value in enumerate(recent_data):
            # 根据指标类型和数值大小决定显示格式
            if thresholds['type'] == 'ratio':
                # 比率类指标显示为百分比
                display_value = value * 100
                if abs(display_value) < 1:
                    value_str = f"{display_value:.4f}%"
                elif abs(display_value) < 10:
                    value_str = f"{display_value:.3f}%"
                else:
                    value_str = f"{display_value:.2f}%"
            else:
                # 数值类指标不显示百分比
                if abs(value) < 0.01:
                    value_str = f"{value:.4f}"
                elif abs(value) < 1:
                    value_str = f"{value:.3f}"
                else:
                    value_str = f"{value:.2f}"
            y_offset = 10 if value > np.mean(recent_data) else -15
            
            plt.annotate(value_str,
                        xy=(dates[i], value),
                        xytext=(0, y_offset),
                        textcoords='offset points',
                        ha='center',
                        va='bottom' if value > np.mean(recent_data) else 'top',
                        fontsize=9,
                        color='#228be6',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='#228be6', alpha=0.8))
        
        # 动态设置y轴范围
        data_range = max(recent_data) - min(recent_data)
        if data_range == 0:  # 处理所有值相同的情况
            margin = abs(max(recent_data)) * 0.1 if max(recent_data) != 0 else 0.1
            y_min = min(recent_data) - margin
            y_max = max(recent_data) + margin
        else:
            # 根据数据范围动态计算边界余量
            margin_ratio = 0.2  # 设置20%的边界余量
            y_min = min(min(recent_data), lower) - data_range * margin_ratio
            y_max = max(max(recent_data), upper) + data_range * margin_ratio
            
            # 确保y轴范围不会太小
            if abs(y_max - y_min) < 1e-10:
                y_min -= 0.1
                y_max += 0.1
        
        plt.ylim(y_min, y_max)
        
        # 美化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
    
    # 设置标题和标签
    plt.title(f'{feature}近7天趋势', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('指标值', fontsize=11, labelpad=10)
    plt.xlabel('日期', fontsize=11, labelpad=10)
    plt.xticks(rotation=30, ha='right')
    
    # 优化图例
    legend = plt.legend(loc='upper right', frameon=True, fancybox=True,
                       shadow=False, fontsize=9)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('#dddddd')
    
    # 转换图表为base64编码
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode()

def plot_indicator_distribution(data, feature, warning_thresholds):
    """生成指标分布箱线图"""
    plt.figure(figsize=(10, 4))
    
    # 获取历史数据和当前值
    historical_data = data[feature].iloc[:-1]
    current_value = data[feature].iloc[-1]
    
    # 绘制箱线图
    plt.boxplot(historical_data, positions=[1], widths=0.7)
    
    # 添加当前值的散点
    plt.plot(1, current_value, 'ro', markersize=10, label='当前值')
    
    if feature in warning_thresholds:
        thresholds = warning_thresholds[feature]
        upper = thresholds['upper']
        lower = thresholds['lower']
        
        # 添加预警阈值线
        plt.axhline(y=upper, color='r', linestyle='--', label='上限阈值')
        plt.axhline(y=lower, color='r', linestyle='--', label='下限阈值')
    
    plt.title(f'{feature}分布情况')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 调整x轴
    plt.xticks([1], ['历史分布'])
    
    # 转换图表为base64编码
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode()

def generate_risk_conclusion(data, warning_thresholds, top_features):
    """生成风险监测结论"""
    conclusion = "<div class='section'><h2>风险监测结论</h2>"
    conclusion += "<div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
    
    # 分析当天异常指标
    current_anomalies = []
    for feature in top_features:
        if feature in data.columns and feature in warning_thresholds:
            current_value = data[feature].iloc[-1]
            thresholds = warning_thresholds[feature]
            if current_value > thresholds['upper']:
                current_anomalies.append(f"{feature}（{current_value:.2f}）高于上限阈值（{thresholds['upper']:.2f}）")
            elif current_value < thresholds['lower']:
                current_anomalies.append(f"{feature}（{current_value:.2f}）低于下限阈值（{thresholds['lower']:.2f}）")
    
    # 分析近7天趋势
    trend_analysis = []
    for feature in top_features:
        if feature in data.columns:
            recent_data = data[feature].iloc[-7:]
            consecutive_anomalies = 0
            trend_direction = ''
            
            # 检查连续异常
            if feature in warning_thresholds:
                thresholds = warning_thresholds[feature]
                for value in recent_data:
                    if value > thresholds['upper'] or value < thresholds['lower']:
                        consecutive_anomalies += 1
            
            # 判断趋势方向
            if len(recent_data) >= 3:
                if all(recent_data.iloc[i] < recent_data.iloc[i+1] for i in range(len(recent_data)-1)):
                    trend_direction = '持续上升'
                elif all(recent_data.iloc[i] > recent_data.iloc[i+1] for i in range(len(recent_data)-1)):
                    trend_direction = '持续下降'
            
            if consecutive_anomalies >= 3:
                trend_analysis.append(f"{feature}连续{consecutive_anomalies}天异常")
            if trend_direction:
                trend_analysis.append(f"{feature}呈{trend_direction}趋势")
    
    # 生成结论文本
    if current_anomalies:
        conclusion += "<p><strong>当前异常指标：</strong></p>"
        conclusion += "<ul style='color: #e74c3c;'>"
        for anomaly in current_anomalies:
            conclusion += f"<li>{anomaly}</li>"
        conclusion += "</ul>"
    
    if trend_analysis:
        conclusion += "<p><strong>近期风险趋势：</strong></p>"
        conclusion += "<ul style='color: #e67e22;'>"
        for trend in trend_analysis:
            conclusion += f"<li>{trend}</li>"
        conclusion += "</ul>"
    
    if not current_anomalies and not trend_analysis:
        conclusion += "<p style='color: #27ae60;'><strong>当前各项指标正常，未发现明显风险。</strong></p>"
    else:
        conclusion += "<p><strong>风险提示：</strong></p>"
        conclusion += "<p style='color: #c0392b;'>建议关注异常指标的变化情况，及时采取风险控制措施。</p>"
    
    conclusion += "</div></div>"
    return conclusion

def generate_monitoring_report(data, warning_thresholds, top_features, warnings=None, anomalies=None):
    """生成风险监测报告"""
    # 创建HTML报告
    report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>风险监测报告</title>
        <style>
            body {{ font-family: SimHei, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
            h1, h2 {{ color: #2c3e50; font-size: 24px; text-align: center; margin-bottom: 10px; }}
            h2 {{ font-size: 20px; margin-top: 30px; }}
            .timestamp {{ color: #7f8c8d; text-align: center; margin-bottom: 30px; }}
            .indicator-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .indicator-table th, .indicator-table td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            .indicator-table th {{ background-color: #f2f2f2; font-weight: bold; }}
            .abnormal {{ color: #e74c3c; font-weight: bold; }}
            .normal {{ color: #27ae60; font-weight: bold; }}
            .trend-charts {{ display: flex; flex-wrap: wrap; justify-content: space-around; gap: 20px; margin-top: 30px; }}
            .chart-container {{ width: 45%; min-width: 300px; margin-bottom: 20px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .section {{ margin-bottom: 40px; }}
            ul {{ padding-left: 20px; margin: 10px 0; }}
            li {{ margin: 5px 0; }}
        </style>
    </head>
    <body>
        <h1>风险监测报告</h1>
        <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # 添加指标状态表格
    report += generate_indicator_table(data, warning_thresholds, top_features)
    
    # 添加近7天数据展示
    report += generate_weekly_data_table(data, warning_thresholds, top_features)
    
    # 添加趋势图表和分布图表
    report += "\n<h2>指标分析</h2>\n<div class='trend-charts'>"
    for feature in top_features:
        if feature in data.columns:
            trend_chart = plot_indicator_trend(data, feature, warning_thresholds)
            dist_chart = plot_indicator_distribution(data, feature, warning_thresholds)
            report += f"""
            <div class='chart-container'>
                <img src="data:image/png;base64,{trend_chart}" style="width:100%;">
            </div>
            <div class='chart-container'>
                <img src="data:image/png;base64,{dist_chart}" style="width:100%;">
            </div>
            """
    
    report += "</div>"
    
    # 添加风险监测结论
    report += generate_risk_conclusion(data, warning_thresholds, top_features)
    
    report += "\n</body>\n</html>"
    
    # 保存报告
    report_path = 'risk_monitoring_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path

def generate_weekly_data_table(data, warning_thresholds, top_features):
    """生成近7天数据表格HTML"""
    table_html = """
        <div class="section">
        <h2>近7天数据展示</h2>
        <table class="indicator-table">
            <thead>
                <tr>
                    <th>监测指标</th>
                    <th>Day-7</th>
                    <th>Day-6</th>
                    <th>Day-5</th>
                    <th>Day-4</th>
                    <th>Day-3</th>
                    <th>Day-2</th>
                    <th>Day-1</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for feature in top_features:
        if feature in data.columns:
            recent_data = data[feature].iloc[-7:]
            is_ratio = any(keyword in feature.lower() for keyword in ['比', '率', '占比', '集中度'])
            
            table_html += f"<tr><td>{feature}</td>"
            
            # 获取预警阈值
            thresholds = warning_thresholds.get(feature, {})
            upper = thresholds.get('upper', float('inf'))
            lower = thresholds.get('lower', float('-inf'))
            
            for value in recent_data:
                # 判断状态
                status_class = ""
                if value > upper or value < lower:
                    status_class = "abnormal"
                
                # 格式化显示值
                if is_ratio:
                    value_display = f"{value:.2%}"
                else:
                    value_display = f"{value:,.2f}"
                
                table_html += f"<td class=\"{status_class}\">{value_display}</td>"
            
            table_html += "</tr>\n"
    
    table_html += "</tbody></table></div>"
    return table_html

def generate_indicator_table(data, warning_thresholds, top_features):
    """生成指标状态表格HTML"""
    table_html = """
        <h2>当前指标状态</h2>
        <table class="indicator-table">
            <thead>
                <tr>
                    <th>监测指标</th>
                    <th>当前值</th>
                    <th>预警阈值</th>
                    <th>超出预警程度</th>
                    <th>状态</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # 创建FeatureSelection实例以获取指标公式
    feature_selector = FeatureSelection()
    
    for feature in top_features:
        if feature in data.columns:
            current_value = data[feature].iloc[-1]
            
            # 获取指标计算公式
            formula = feature_selector.get_indicator_formula(feature)
            
            # 判断是否为比率类指标
            is_ratio = any(keyword in feature.lower() for keyword in ['比', '率', '占比', '集中度'])
            
            # 获取预警阈值
            if feature in warning_thresholds:
                thresholds = warning_thresholds[feature]
                threshold_display = f"{thresholds['lower']:.2%} ~ {thresholds['upper']:.2%}"
                if is_ratio:
                    threshold_display = f"{thresholds['lower']:.2%} ~ {thresholds['upper']:.2%}"
                else:
                    threshold_display = f"{thresholds['lower']:,.2f} ~ {thresholds['upper']:,.2f}"
                
                # 计算超出预警阈值的百分比
                if current_value > thresholds['upper']:
                    deviation = (current_value - thresholds['upper']) / thresholds['upper'] * 100
                    deviation_display = f"+{deviation:.1f}%"
                elif current_value < thresholds['lower']:
                    deviation = (current_value - thresholds['lower']) / thresholds['lower'] * 100
                    deviation_display = f"-{abs(deviation):.1f}%"
                else:
                    deviation_display = "0.0%"
            else:
                threshold_display = "-"
                deviation_display = "-"
            
            # 判断状态
            status = "正常"
            status_class = "normal"
            if feature in warning_thresholds:
                if current_value > warning_thresholds[feature]['upper'] or \
                   current_value < warning_thresholds[feature]['lower']:
                    status = "异常"
                    status_class = "abnormal"
            
            # 格式化显示值
            if is_ratio:
                value_display = f"{current_value:.2%}"
            else:
                value_display = f"{current_value:,.2f}"
            
            table_html += f"""
                <tr>
                    <td>{feature}<br><span style="font-size: 0.9em; color: #666;">({formula})</span></td>
                    <td class="{status_class if status == '异常' else ''}">{value_display}</td>
                    <td>{threshold_display}</td>
                    <td class="{status_class if status == '异常' else ''}">{deviation_display}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
            """
    
    table_html += "</tbody></table>"
    return table_html

def generate_indicator_table(data, warning_thresholds, top_features):
    """生成指标状态表格HTML"""
    table_html = """
        <h2>当前指标状态</h2>
        <table class="indicator-table">
            <thead>
                <tr>
                    <th>监测指标</th>
                    <th>当前值</th>
                    <th>预警阈值</th>
                    <th>超出预警程度</th>
                    <th>状态</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for feature in top_features:
        if feature in data.columns:
            current_value = data[feature].iloc[-1]
            prev_value = data[feature].iloc[-2] if len(data[feature]) > 1 else current_value
            
            # 判断是否为比率类指标
            is_ratio = any(keyword in feature.lower() for keyword in ['比', '率', '占比', '集中度'])
            
            # 获取预警阈值
            if feature in warning_thresholds:
                thresholds = warning_thresholds[feature]
                if is_ratio:
                    threshold_display = f"{thresholds['lower']:.2%} ~ {thresholds['upper']:.2%}"
                else:
                    threshold_display = f"{thresholds['lower']:,.2f} ~ {thresholds['upper']:,.2f}"
                
                # 计算超出预警阈值的百分比
                if current_value > thresholds['upper']:
                    deviation = (current_value - thresholds['upper']) / thresholds['upper'] * 100
                    deviation_display = f"↑{deviation:.1f}%"
                    trend_class = "abnormal"
                elif current_value < thresholds['lower']:
                    deviation = (current_value - thresholds['lower']) / thresholds['lower'] * 100
                    deviation_display = f"↓{deviation:.1f}%"
                    trend_class = "abnormal"
                else:
                    deviation_display = "→"
                    trend_class = ""
            else:
                threshold_display = "-"
                deviation_display = "-"
                trend_class = ""
            
            # 判断状态
            status = "正常"
            status_class = "normal"
            if feature in warning_thresholds:
                if current_value > warning_thresholds[feature]['upper'] or \
                   current_value < warning_thresholds[feature]['lower']:
                    status = "异常"
                    status_class = "abnormal"
            
            # 格式化显示值
            if is_ratio:
                value_display = f"{current_value:.2%}"
            else:
                value_display = f"{current_value:,.2f}"
            
            table_html += f"""
                <tr>
                    <td>{feature}</td>
                    <td class="{status_class if status == '异常' else ''}">{value_display}</td>
                    <td>{threshold_display}</td>
                    <td class="{trend_class}">{deviation_display}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
            """
    
    table_html += "</tbody></table>"
    return table_html