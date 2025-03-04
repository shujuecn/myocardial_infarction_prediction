# data_generation.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 设置随机种子以确保可重复性
np.random.seed(42)


def generate_clinical_data(n_samples=5000):
    """
    生成心肌梗死预后相关的临床数据

    参数:
        n_samples: 样本数量

    返回:
        X: 特征数据
        y: 目标变量（0: 良好预后, 1: 不良预后）
    """
    # 生成基础变量
    age = np.random.normal(65, 10, n_samples)  # 年龄，均值65，标准差10
    age = np.clip(age, 30, 95)  # 限制年龄范围

    # 性别 (0: 女性, 1: 男性)
    gender = np.random.binomial(1, 0.6, n_samples)  # 男性占比60%

    # 甘油三酯 (正态分布，单位: mmol/L)
    triglycerides = np.random.lognormal(0.5, 0.4, n_samples)
    triglycerides = np.clip(triglycerides, 0.3, 5.0)

    # 空腹血糖 (正态分布，单位: mmol/L)
    fasting_glucose = np.random.normal(5.5, 1.2, n_samples)
    fasting_glucose = np.clip(fasting_glucose, 3.5, 15.0)

    # 计算TyG指数 = ln[triglycerides (mg/dL) × fasting glucose (mg/dL)/2]
    # 先将单位转换: mmol/L -> mg/dL
    triglycerides_mgdl = triglycerides * 88.57
    glucose_mgdl = fasting_glucose * 18.02
    tyg_index = np.log(triglycerides_mgdl * glucose_mgdl / 2)

    # 其他临床变量
    systolic_bp = np.random.normal(130, 20, n_samples)  # 收缩压
    systolic_bp = np.clip(systolic_bp, 90, 200)

    diastolic_bp = np.random.normal(80, 15, n_samples)  # 舒张压
    diastolic_bp = np.clip(diastolic_bp, 50, 120)

    heart_rate = np.random.normal(75, 15, n_samples)  # 心率
    heart_rate = np.clip(heart_rate, 40, 150)

    ldl_c = np.random.normal(3.0, 1.0, n_samples)  # 低密度脂蛋白胆固醇
    ldl_c = np.clip(ldl_c, 0.5, 7.0)

    hdl_c = np.random.normal(1.2, 0.4, n_samples)  # 高密度脂蛋白胆固醇
    hdl_c = np.clip(hdl_c, 0.3, 3.0)

    creatinine = np.random.lognormal(-0.1, 0.3, n_samples)  # 肌酐
    creatinine = np.clip(creatinine, 30, 200)

    hba1c = np.random.normal(6.0, 1.2, n_samples)  # 糖化血红蛋白
    hba1c = np.clip(hba1c, 4.0, 15.0)

    bmi = np.random.normal(25, 4, n_samples)  # 体重指数
    bmi = np.clip(bmi, 15, 45)

    smoking = np.random.binomial(1, 0.3, n_samples)  # 吸烟状态

    # 疾病史 (0: 无, 1: 有)
    hypertension = np.random.binomial(1, 0.4, n_samples)  # 高血压病史
    diabetes = np.random.binomial(1, 0.25, n_samples)  # 糖尿病病史
    previous_mi = np.random.binomial(1, 0.15, n_samples)  # 既往心肌梗死史

    # 将TyG指数作为核心变量，与预后密切相关
    # 生成预后标签（0: 良好预后, 1: 不良预后）
    # 构建一个包含多变量的模型来确定预后概率
    logit = (
        -10
        + 0.05 * age
        + 0.3 * gender
        + 1.8 * tyg_index
        + 0.01 * systolic_bp
        + 0.2 * ldl_c
        - 0.3 * hdl_c
        + 0.4 * diabetes
        + 0.5 * previous_mi
        + 0.02 * heart_rate
        + 0.01 * creatinine
        + 0.3 * smoking
        + 0.2 * hba1c
        + 0.05 * bmi
        - 0.0001 * (tyg_index - 8.5) ** 2
    )  # 非线性影响

    prob = 1 / (1 + np.exp(-logit))  # sigmoid函数转换为概率
    prob = np.clip(prob, 0.1, 0.8)
    y = np.random.binomial(1, prob)  # 基于概率生成二分类标签

    # 创建特征DataFrame
    data = {
        "Age": age,
        "Gender": gender,
        "TyG_Index": tyg_index,
        "Triglycerides": triglycerides,
        "Fasting_Glucose": fasting_glucose,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "Heart_Rate": heart_rate,
        "LDL_C": ldl_c,
        "HDL_C": hdl_c,
        "Creatinine": creatinine,
        "HbA1c": hba1c,
        "BMI": bmi,
        "Smoking": smoking,
        "Hypertension": hypertension,
        "Diabetes": diabetes,
        "Previous_MI": previous_mi,
    }

    X = pd.DataFrame(data)

    return X, y


def split_save_data():
    """
    生成数据，划分训练集和测试集，并保存到CSV文件
    """
    X, y = generate_clinical_data(n_samples=5000)

    # 添加标签到数据中
    data = X.copy()
    data["Poor_Prognosis"] = y

    # 保存完整数据集
    data.to_csv("./data/mi_clinical_data.csv", index=False)

    # 划分训练集和测试集 (70% 训练, 30% 测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 保存训练集和测试集
    train_data = X_train.copy()
    train_data["Poor_Prognosis"] = y_train
    train_data.to_csv("./data/train_data.csv", index=False)

    test_data = X_test.copy()
    test_data["Poor_Prognosis"] = y_test
    test_data.to_csv("./data/test_data.csv", index=False)

    print("数据已生成并保存:")
    print(f"- 全部数据: {len(data)} 行")
    print(f"- 训练集: {len(train_data)} 行")
    print(f"- 测试集: {len(test_data)} 行")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    os.makedirs("./data/", exist_ok=True)
    split_save_data()
    print("数据生成完成！")
