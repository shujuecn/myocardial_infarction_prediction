# model_training.py
import os
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

font_path = "./times+simsun.ttf"
zh_font = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = zh_font.get_name()
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


def load_data():
    """加载训练和测试数据"""
    train_data = pd.read_csv("./data/train_data.csv")
    test_data = pd.read_csv("./data/test_data.csv")

    X_train = train_data.drop("Poor_Prognosis", axis=1)
    y_train = train_data["Poor_Prognosis"]

    X_test = test_data.drop("Poor_Prognosis", axis=1)
    y_test = test_data["Poor_Prognosis"]

    return X_train, X_test, y_train, y_test


# def feature_selection_lasso(X_train, y_train, X_test):
#     """使用LASSO进行特征选择"""
#     # 标准化特征
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # LASSO特征选择
#     # lasso = LassoCV(cv=5, random_state=42, max_iter=50)
#     lasso = LassoCV(cv=10, random_state=42, max_iter=500, alphas=np.logspace(-4, 1, 50))
#     lasso.fit(X_train_scaled, y_train)

#     # 获取特征重要性
#     feature_importance = np.abs(lasso.coef_)
#     feature_names = X_train.columns

#     # 特征重要性排序
#     feature_importance_df = pd.DataFrame(
#         {"Feature": feature_names, "Importance": feature_importance}
#     ).sort_values(by="Importance", ascending=False)

#     # 选择非零系数的特征
#     selected_features = feature_names[feature_importance > 0]

#     print(f"LASSO特征选择: {len(selected_features)}/{len(feature_names)} 个特征被选中")
#     print("选中的特征:")
#     print(feature_importance_df[feature_importance_df["Importance"] > 0])

#     # 确保TyG_Index被选中（因为它是我们的核心变量）
#     if "TyG_Index" not in selected_features:
#         selected_features = selected_features.append(pd.Index(["TyG_Index"]))
#         print("注意: TyG_Index被手动添加为核心变量")

#     # 返回筛选后的数据
#     return (
#         X_train[selected_features],
#         X_test[selected_features],
#         selected_features,
#         scaler,
#     )


def tree_based_feature_selection(X_train, y_train, X_test, num_features=10):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    feature_importances = model.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    selected_features = importance_df["Feature"].iloc[:num_features]

    return (
        X_train[selected_features],
        X_test[selected_features],
        selected_features,
        scaler,
    )


def train_xgboost_model(X_train, y_train):
    """使用网格搜索训练XGBoost模型"""
    # 定义参数网格
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    # 使用较小的参数网格进行快速测试
    small_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.2],
    }

    # 初始化XGBoost分类器
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )

    # 使用5折交叉验证的网格搜索
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=small_param_grid,  # 使用小参数网格加快训练
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
    )

    # 训练模型
    grid_search.fit(X_train, y_train)

    # 获取最佳参数和模型
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("最佳参数:")
    print(best_params)
    print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

    return best_model


def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """评估模型并生成评估图表"""
    # 创建模型目录和图表目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # 训练集和测试集的预测
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 交叉验证预测
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_cv_pred_proba = cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    # 保存模型
    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # 保存特征名称
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    # 1. ROC曲线
    plt.figure(figsize=(10, 8))

    # 训练集ROC
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
    auc_train = roc_auc_score(y_train, y_train_pred_proba)
    plt.plot(fpr_train, tpr_train, label=f"训练集 (AUC = {auc_train:.3f})")

    # 交叉验证ROC
    fpr_cv, tpr_cv, _ = roc_curve(y_train, y_cv_pred_proba)
    auc_cv = roc_auc_score(y_train, y_cv_pred_proba)
    plt.plot(fpr_cv, tpr_cv, label=f"5折交叉验证 (AUC = {auc_cv:.3f})")

    # 测试集ROC
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
    auc_test = roc_auc_score(y_test, y_test_pred_proba)
    plt.plot(fpr_test, tpr_test, label=f"测试集 (AUC = {auc_test:.3f})")

    # 添加对角线
    plt.plot([0, 1], [0, 1], "k--")

    plt.xlabel("假阳性率 (FPR)")
    plt.ylabel("真阳性率 (TPR)")
    plt.title("XGBoost模型的ROC曲线")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/roc_curve.png", dpi=300, bbox_inches="tight")

    # 2. 校准曲线
    plt.figure(figsize=(10, 8))

    # 训练集校准曲线
    prob_true_train, prob_pred_train = calibration_curve(
        y_train, y_train_pred_proba, n_bins=10
    )
    brier_train = brier_score_loss(y_train, y_train_pred_proba)
    plt.plot(
        prob_pred_train,
        prob_true_train,
        marker="o",
        linewidth=1,
        label=f"训练集 (Brier = {brier_train:.3f})",
    )

    # 测试集校准曲线
    prob_true_test, prob_pred_test = calibration_curve(
        y_test, y_test_pred_proba, n_bins=10
    )
    brier_test = brier_score_loss(y_test, y_test_pred_proba)
    plt.plot(
        prob_pred_test,
        prob_true_test,
        marker="s",
        linewidth=1,
        label=f"测试集 (Brier = {brier_test:.3f})",
    )

    # 完美校准
    plt.plot([0, 1], [0, 1], "k--", label="完美校准")

    plt.xlabel("预测概率")
    plt.ylabel("观察到的频率")
    plt.title("XGBoost模型的校准曲线")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/calibration_curve.png", dpi=300, bbox_inches="tight")

    # 3. 决策曲线分析 (DCA)
    def calculate_net_benefit(threshold, y_true, y_pred_proba):
        y_pred = (y_pred_proba >= threshold).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)

        # 如果没有预测为阳性的病例，则净收益为0
        if TP + FP == 0:
            return 0

        # 计算净收益
        net_benefit = (TP / n) - (FP / n) * (threshold / (1 - threshold))
        return net_benefit

    # 计算"全部患者接受干预"策略的净收益
    def calculate_all_net_benefit(threshold, y_true):
        n = len(y_true)
        prevalence = np.sum(y_true) / n
        return prevalence - (1 - prevalence) * threshold / (1 - threshold)

    thresholds = np.arange(0.01, 0.99, 0.01)

    # 计算XGBoost模型的净收益
    nb_test = [calculate_net_benefit(t, y_test, y_test_pred_proba) for t in thresholds]

    # 计算"全部患者接受干预"的净收益
    nb_all = [calculate_all_net_benefit(t, y_test) for t in thresholds]

    # "无患者接受干预"的净收益恒为0
    nb_none = np.zeros_like(thresholds)

    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, nb_test, label="XGBoost模型")
    plt.plot(thresholds, nb_all, label="全部患者接受干预")
    plt.plot(thresholds, nb_none, label="无患者接受干预")
    plt.xlabel("风险阈值")
    plt.ylabel("净收益")
    plt.title("决策曲线分析 (DCA)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/decision_curve.png", dpi=300, bbox_inches="tight")

    # 4. 混淆矩阵与评估指标
    cm_test = confusion_matrix(y_test, y_test_pred)

    # 计算各项评估指标
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    specificity = cm_test[0, 0] / (cm_test[0, 0] + cm_test[0, 1])
    npv = (
        cm_test[0, 0] / (cm_test[0, 0] + cm_test[1, 0])
        if (cm_test[0, 0] + cm_test[1, 0]) > 0
        else 0
    )

    # 计算阳性似然比和阴性似然比
    plr = recall / (1 - specificity) if (1 - specificity) > 0 else float("inf")
    nlr = (1 - recall) / specificity if specificity > 0 else float("inf")

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_test,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["预测良好预后", "预测不良预后"],
        yticklabels=["实际良好预后", "实际不良预后"],
    )
    plt.ylabel("实际标签")
    plt.xlabel("预测标签")
    plt.title("测试集混淆矩阵")
    plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches="tight")

    # 打印评估指标
    print("\n模型评估指标:")
    print(f"AUC (训练集): {auc_train:.4f}")
    print(f"AUC (交叉验证): {auc_cv:.4f}")
    print(f"AUC (测试集): {auc_test:.4f}")
    print(f"Brier Score (测试集): {brier_test:.4f}")

    print("\n混淆矩阵评估指标:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率/敏感性 (Recall/Sensitivity): {recall:.4f}")
    print(f"特异性 (Specificity): {specificity:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"阴性预测值 (NPV): {npv:.4f}")
    print(f"阳性似然比 (PLR): {plr:.4f}")
    print(f"阴性似然比 (NLR): {nlr:.4f}")

    # 保存评估指标
    metrics = {
        "AUC_Train": auc_train,
        "AUC_CV": auc_cv,
        "AUC_Test": auc_test,
        "Brier_Test": brier_test,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1_Score": f1,
        "NPV": npv,
        "PLR": plr,
        "NLR": nlr,
    }

    with open("models/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    return auc_test, metrics


def main():
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_data()
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 2. 特征选择
    X_train_selected, X_test_selected, selected_features, scaler = (
        tree_based_feature_selection(X_train, y_train, X_test)
    )
    print(
        f"特征选择后 - 训练集: {X_train_selected.shape}, 测试集: {X_test_selected.shape}"
    )

    # 保存选中特征和标准化对象
    joblib.dump(selected_features, "models/selected_features.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # 3. 训练XGBoost模型
    model = train_xgboost_model(X_train_selected, y_train)

    # 4. 评估模型
    auc, metrics = evaluate_model(
        model, X_train_selected, y_train, X_test_selected, y_test, selected_features
    )

    print("\n模型训练与评估完成！")
    print(f"测试集AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
