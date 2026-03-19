import pandas as pd
import requests
import time
from urllib.parse import quote
import json

def classify_smiles_from_csv(input_csv, output_csv, smiles_column='smiles', use_cache=True, delay=0.1):
    """
    从CSV读取SMILES，调用分类API，结果保存到新CSV
    
    参数:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        smiles_column: SMILES列名
        use_cache: 是否使用缓存
        delay: 请求间隔(秒)，避免频繁请求
    """
    
    # 读取CSV文件
    try:
        df = pd.read_csv(input_csv)
        print(f"成功读取文件: {input_csv}")
        print(f"数据行数: {len(df)}")
        print(f"列名: {list(df.columns)}")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 检查SMILES列是否存在
    if smiles_column not in df.columns:
        print(f"错误: 未找到列 '{smiles_column}'")
        print(f"可用列: {list(df.columns)}")
        return
    
    # 准备结果列表
    classification_results = []
    successful_count = 0
    
    # 遍历每个SMILES
    for index, smiles in enumerate(df[smiles_column]):
        if pd.isna(smiles) or smiles == '':
            print(f"第 {index+1} 行: 空SMILES，跳过")
            classification_results.append({"error": "Empty SMILES"})
            continue
        
        print(f"处理第 {index+1}/{len(df)} 行: {smiles}")
        
        try:
            # 调用API
            encoded_smiles = quote(str(smiles))
            url = f"https://npclassifier.ucsd.edu/classify?smiles={encoded_smiles}"
            if use_cache:
                url += "&cached=true"
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                classification_results.append(result)
                successful_count += 1
                
                # 显示分类结果
                class_results = result.get('class_results', [])
                if class_results:
                    print(f"  成功 - 类别: {class_results[0]}")
                else:
                    print(f"  成功 - 无分类结果")
            else:
                error_msg = f"API错误: {response.status_code}"
                print(f"  失败: {error_msg}")
                classification_results.append({"error": error_msg, "smiles": smiles})
                
        except Exception as e:
            error_msg = f"请求异常: {str(e)}"
            print(f"  异常: {error_msg}")
            classification_results.append({"error": error_msg, "smiles": smiles})
        
        # 延迟，避免频繁请求
        time.sleep(delay)
    
    print(f"\n处理完成! 成功: {successful_count}/{len(df)}")
    
    # 将结果转换为DataFrame并展开嵌套结构
    results_df = process_classification_results(classification_results)
    
    # 合并原始数据和分类结果
    final_df = pd.concat([df, results_df], axis=1)
    
    # 保存到CSV
    try:
        final_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n结果已保存到: {output_csv}")
        print(f"总行数: {len(final_df)}")
        
        # 显示一些统计信息
        if 'class_results_primary' in final_df.columns:
            class_counts = final_df['class_results_primary'].value_counts()
            print(f"\n主要类别分布 (前10):")
            for class_name, count in class_counts.head(10).items():
                print(f"  {class_name}: {count}")
                
    except Exception as e:
        print(f"保存CSV文件失败: {e}")

def process_classification_results(results):
    """
    处理NPClassifier API返回结果
    根据实际响应结构调整字段提取
    """
    processed_data = []
    
    for result in results:
        row_data = {}
        
        # 错误信息
        row_data['api_error'] = result.get('error', '')
        
        # 主要分类结果（取第一个作为主要分类）
        class_results = result.get('class_results', [])
        row_data['class_results_primary'] = class_results[0] if class_results else ''
        row_data['class_results_all'] = ';'.join(class_results) if class_results else ''
        row_data['class_results_count'] = len(class_results)
        
        # 超类结果
        superclass_results = result.get('superclass_results', [])
        row_data['superclass_results_primary'] = superclass_results[0] if superclass_results else ''
        row_data['superclass_results_all'] = ';'.join(superclass_results) if superclass_results else ''
        row_data['superclass_results_count'] = len(superclass_results)
        
        # 通路结果
        pathway_results = result.get('pathway_results', [])
        row_data['pathway_results_primary'] = pathway_results[0] if pathway_results else ''
        row_data['pathway_results_all'] = ';'.join(pathway_results) if pathway_results else ''
        row_data['pathway_results_count'] = len(pathway_results)
        
        # 其他字段
        row_data['isglycoside'] = result.get('isglycoside', False)
        
        # 不记录fp指纹数据
        
        processed_data.append(row_data)
    
    return pd.DataFrame(processed_data)

# 使用示例
if __name__ == "__main__":
    # 基本用法
    classify_smiles_from_csv(
        input_csv='results/FlowNP_pocket.csv',
        output_csv='data/classified_FlowNP_pocket.csv',
        smiles_column='smiles',
        use_cache=True,
        delay=0.02  # 20ms间隔
    )