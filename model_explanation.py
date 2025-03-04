# model_explanation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import shap
import os
import joblib
from matplotlib import font_manager


font_path = "./times+simsun.ttf"
zh_font = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = zh_font.get_name()
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


def load_model_and_data():
    """加载模型、特征和数据"""
    # 加载模型
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 加载特征名称
    selected_features = joblib.load("models/selected_features.pkl")

    # 加载数据
    test_data = pd.read_csv("test_data.csv")
    X_test = test_data[selected_features]
    y_test = test_data["Poor_Prognosis"]

    return model, X_test, y_test, selected_features


def explain_model_with_shap(model, X):
    """使用SHAP进行模型解释"""
    # 创建SHAP图像目录
    os.makedirs("plots/shap", exist_ok=True)

    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)

    # 计算SHAP值
    # 如果数据集太大，可以随机抽样一部分数据
    if len(X) > 500:
        sample_indices = np.random.choice(len(X), 500, replace=False)
        X_sample = X.iloc[sample_indices]
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)

    # SHAP值保存，供Web应用使用
    with open("models/shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    # 1. 汇总图 - 显示所有特征的整体重要性
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("特征SHAP值汇总图")
    plt.tight_layout()
    plt.savefig("plots/shap/summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. 条形图 - 特征重要性排序
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("特征重要性条形图")
    plt.tight_layout()
    plt.savefig("plots/shap/feature_importance_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. 特定特征的依赖图 - 重点是TyG_Index
    if "TyG_Index" in X.columns:
        plt.figure(figsize=(10, 8))
        shap.dependence_plot("TyG_Index", shap_values, X_sample, show=False)
        plt.title("TyG指数的SHAP依赖图")
        plt.tight_layout()
        plt.savefig("plots/shap/tyg_dependence_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 4. 瀑布图 - 随机选择一个样本进行详细解释
    # 随机选择一个样例
    # sample_idx = np.random.choice(len(X_sample))
    # plt.figure(figsize=(12, 8))
    # shap.waterfall_plot(
    #     shap.Explanation(
    #         values=shap_values[sample_idx],
    #         base_values=explainer.expected_value,
    #         data=X_sample.iloc[sample_idx],
    #     )
    # )
    # plt.title(f"样本 #{sample_idx} 的SHAP瀑布图")
    # plt.tight_layout()
    # plt.savefig("plots/shap/waterfall_plot.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # 5. 力图 - 为几个样本创建力图
    # 选择几个有代表性的样本
    sample_indices = np.random.choice(
        len(X_sample), min(5, len(X_sample)), replace=False
    )
    for i, idx in enumerate(sample_indices):
        plt.figure(figsize=(12, 4))
        shap.force_plot(
            explainer.expected_value,
            shap_values[idx],
            X_sample.iloc[idx],
            matplotlib=True,
            show=False,
        )
        plt.title(f"样本 #{idx} 的SHAP力图")
        plt.tight_layout()
        plt.savefig(f"plots/shap/force_plot_{i}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("SHAP解释图已保存至 plots/shap/ 目录")

    return explainer


def main():
    # 加载模型和数据
    model, X_test, y_test, feature_names = load_model_and_data()

    # SHAP解释
    explain_model_with_shap(model, X_test)

    print("模型解释完成!")


if __name__ == "__main__":
    main()
