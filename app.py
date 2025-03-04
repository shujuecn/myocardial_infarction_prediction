# app.py
import base64
import io
import pickle

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template, request
from matplotlib import font_manager

matplotlib.use("Agg")  # 设置 Matplotlib 后端为非 GUI 模式

font_path = "./times+simsun.ttf"
zh_font = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = zh_font.get_name()
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

app = Flask(__name__)


# 加载模型和相关文件
def load_model_files():
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)

    selected_features = joblib.load("models/selected_features.pkl")

    with open("models/metrics.pkl", "rb") as f:
        metrics = pickle.load(f)

    return model, explainer, selected_features, metrics


model, explainer, selected_features, metrics = load_model_files()


# 主页路由
@app.route("/")
def index():
    # 构建特征字典，包含每个特征的类型和可能的范围
    features_info = {
        "Age": {"type": "numeric", "min": 30, "max": 95, "step": 1, "default": 65},
        "Gender": {
            "type": "categorical",
            "options": [{"value": 1, "text": "男性"}, {"value": 0, "text": "女性"}],
            "default": 1,
        },
        "TyG_Index": {
            "type": "numeric",
            "min": 7.5,
            "max": 10.5,
            "step": 0.1,
            "default": 9.0,
        },
        "Triglycerides": {
            "type": "numeric",
            "min": 0.3,
            "max": 5.0,
            "step": 0.1,
            "default": 1.5,
        },
        "Fasting_Glucose": {
            "type": "numeric",
            "min": 3.5,
            "max": 15.0,
            "step": 0.1,
            "default": 5.5,
        },
        "Systolic_BP": {
            "type": "numeric",
            "min": 90,
            "max": 200,
            "step": 1,
            "default": 130,
        },
        "Diastolic_BP": {
            "type": "numeric",
            "min": 50,
            "max": 120,
            "step": 1,
            "default": 80,
        },
        "Heart_Rate": {
            "type": "numeric",
            "min": 40,
            "max": 150,
            "step": 1,
            "default": 75,
        },
        "LDL_C": {
            "type": "numeric",
            "min": 0.5,
            "max": 7.0,
            "step": 0.1,
            "default": 3.0,
        },
        "HDL_C": {
            "type": "numeric",
            "min": 0.3,
            "max": 3.0,
            "step": 0.1,
            "default": 1.2,
        },
        "Creatinine": {
            "type": "numeric",
            "min": 30,
            "max": 200,
            "step": 1,
            "default": 70,
        },
        "HbA1c": {
            "type": "numeric",
            "min": 4.0,
            "max": 15.0,
            "step": 0.1,
            "default": 6.0,
        },
        "BMI": {"type": "numeric", "min": 15, "max": 45, "step": 0.1, "default": 25},
        "Smoking": {
            "type": "categorical",
            "options": [{"value": 1, "text": "是"}, {"value": 0, "text": "否"}],
            "default": 0,
        },
        "Hypertension": {
            "type": "categorical",
            "options": [{"value": 1, "text": "有"}, {"value": 0, "text": "无"}],
            "default": 0,
        },
        "Diabetes": {
            "type": "categorical",
            "options": [{"value": 1, "text": "有"}, {"value": 0, "text": "无"}],
            "default": 0,
        },
        "Previous_MI": {
            "type": "categorical",
            "options": [{"value": 1, "text": "有"}, {"value": 0, "text": "无"}],
            "default": 0,
        },
    }

    # 只保留选中的特征
    selected_features_info = {
        k: features_info[k] for k in selected_features if k in features_info
    }

    return render_template(
        "index.html", features=selected_features_info, metrics=metrics
    )


@app.route("/predict", methods=["POST"])
def predict():
    # 从请求中获取数据
    data = request.json

    # 创建DataFrame
    input_df = pd.DataFrame([data])

    # 确保列顺序与模型训练时一致
    input_df = input_df[selected_features]

    # 预测
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction_label = 1 if prediction_proba >= 0.5 else 0

    # 生成SHAP解释
    shap_values = explainer.shap_values(input_df)

    # 生成SHAP可视化图像
    plt.figure(figsize=(12, 4))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_df.iloc[0],
        feature_names=selected_features,
        matplotlib=True,
        show=False,
    )
    plt.title("特征影响分析")
    plt.tight_layout()

    # 将图像转为base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")

    # 计算各特征贡献度
    total_shap = np.abs(shap_values).sum()
    feature_contributions = {}
    for i, feature in enumerate(selected_features):
        contribution = (
            (np.abs(shap_values[0][i]) / total_shap) * 100 if total_shap > 0 else 0
        )
        feature_contributions[feature] = round(contribution, 2)

    # 特征贡献度排序
    sorted_contributions = sorted(
        feature_contributions.items(), key=lambda x: x[1], reverse=True
    )

    return jsonify(
        {
            "probability": round(float(prediction_proba), 4),
            "label": int(prediction_label),
            "label_text": "不良预后" if prediction_label == 1 else "良好预后",
            "risk_level": get_risk_level(prediction_proba),
            "shap_image": img_str,
            "feature_contributions": sorted_contributions,
        }
    )


def get_risk_level(probability):
    """根据预测概率确定风险等级"""
    if probability < 0.3:
        return {"level": "低风险", "color": "success"}
    elif probability < 0.6:
        return {"level": "中等风险", "color": "warning"}
    else:
        return {"level": "高风险", "color": "danger"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
