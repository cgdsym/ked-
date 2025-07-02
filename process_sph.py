import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import wfdb

import tarfile
import argparse
from sklearn.model_selection import train_test_split


# 定义生成英文多标签描述的函数
def generate_english_labels(X_aha_codes, path):
    """
    生成完整的英文多标签描述
    :param X_aha_codes: 包含编码的数组（一般来自 DataFrame 某列的 values）
    :param path: code.csv 所在的路径
    :return: 生成好的多标签描述列表
    """
    # 读取 code.csv 并构建 code 到描述的映射
    code_csv = pd.read_csv(os.path.join(path, "code.csv"))
    code2text = {}
    for i in range(len(code_csv)):
        code2text[str(code_csv.Code.values[i])] = code_csv.Description.values[i]
    
    texts = []
    for codes in X_aha_codes:
        code_list = codes.split(';')
        text_list = []
        for code in code_list:
            t = " ".join(code2text[c] for c in code.split('+'))
            text_list.append(t)
        texts.append("; ".join(text_list))
    return texts

def split_snomed_codes(x):
    # 支持 , 和 ; 两种分隔符，去除前导0
    if pd.isna(x):
        return []
    codes = re.split(r'[;]', str(x))
    # 去除空字符串和前导0
    return [str(int(code.strip())) for code in codes if code.strip().isdigit() and int(code.strip()) != 0]

def prepare(args):
    path = args.data_dir
    df = pd.read_csv(os.path.join(path, "metadata.csv"))

    # 1. 拿到所有唯一的Patient_ID
    unique_ids = df['Patient_ID'].unique()

    # 2. 划分ID为train和test
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

    # 3. 直接用isin筛选出数据
    df_train = df[df['Patient_ID'].isin(train_ids)].copy()
    df_test = df[df['Patient_ID'].isin(test_ids)].copy()
    #X\Y为拼接为完整信号文件路径
    #训练
    X = df_train.ECG_ID.values
    X = [os.path.join(path, "records/records_wfdb/" + x) for x in X]
    df_train['path']=X
    #测试
    Y = df_test.ECG_ID.values
    Y = [os.path.join(path, "records/records_wfdb/" + y) for y in Y]
    df_test['path']=Y
    #标签
    #训练
    X_aha_codes = df_train.AHA_Code.values 
    X_label = generate_english_labels(X_aha_codes, path)
    #测试
    Y_aha_codes = df_test.AHA_Code.values 
    Y_label = generate_english_labels(Y_aha_codes, path)


    df = pd.DataFrame(X_label, columns=["label"])
    df_train['label']=df['label'].apply(split_snomed_codes)
    print(df_train[['path', 'label']].head(5))

    df = pd.DataFrame(Y_label, columns=["label"])
    df_test['label']=df['label'].apply(split_snomed_codes)
    print(df_test[['path', 'label']].head(20))

    # df_train[['path', 'label']].to_csv(f'/data_C/sdb1/lyi/ECG-Chat-master/data/sph/train.csv',index=False)
    # df_test[['path', 'label']].to_csv(f'/data_C/sdb1/lyi/ECG-Chat-master/data/sph/test.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/data_C/sdb1/lyi/ECG-Chat-master/data/sph/")
    args = parser.parse_args()

    prepare(args)







