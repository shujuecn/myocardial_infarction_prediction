# 心肌梗死预后预测系统

## 介绍

本项目是一个基于 [Claude-3.7-Sonnet](https://claude.ai/) 开发的心肌梗死预后预测系统，基于机器学习算法（XGBoost）实现对心肌梗死患者预后的预测，并通过SHAP可解释性分析提供模型解释。系统还包括一个基于Flask的网页应用，方便用户输入患者信息并获取预测结果。

## 项目结构

```
myocardial_infarction_prediction/
├── data/                      # 保存生成的数据
├── data_generation.py         # 数据生成模块
├── model_training.py          # 模型训练和评估模块
├── model_explanation.py       # 模型可解释性分析模块
├── app.py                     # Flask网页应用
├── templates/                 # HTML模板
│   └── index.html             # 主页面
├── static/                    # 静态文件
│   ├── css/
│   │   └── style.css          # 样式文件
│   └── js/
│       └── main.js            # JavaScript文件
├── models/                    # 保存训练好的模型
│   └── xgb_model.pkl          # XGBoost模型
└── requirements.txt           # 项目依赖
```

## 模块说明

### 1. data_generation.py

该模块用于生成心肌梗死预后相关的临床数据。通过 `generate_clinical_data` 函数生成特征数据和目标变量（预后标签），并使用 `split_save_data` 函数将数据划分为训练集和测试集，保存为CSV文件。

### 2. model_training.py

该模块负责模型的训练和评估。主要步骤包括：

1. 加载数据：从CSV文件中读取训练集和测试集。
2. 特征选择：使用LASSO回归进行特征选择，筛选出对预后预测有显著影响的特征。
3. 模型训练：使用XGBoost分类器，通过网格搜索和交叉验证确定最佳参数，训练得到最优模型（由于LASSO未筛选出任何特征，修改为基于决策树的特征重要性）。
4. 模型评估：对训练好的模型进行评估，生成ROC曲线、校准曲线、混淆矩阵等评估图表，并计算各项评估指标。

### 3. model_explanation.py

该模块使用SHAP（SHapley Additive exPlanations）对模型进行可解释性分析。通过SHAP值分析每个特征对模型预测结果的贡献，生成汇总图、条形图、依赖图、瀑布图和力图等可视化图表，帮助理解模型的决策过程。

### 4. app.py

基于Flask框架的网页应用。用户可以通过网页输入患者的相关临床数据，系统将调用训练好的XGBoost模型进行预测，并返回预测结果和SHAP解释图。同时，网页还展示了模型的评估指标和特征贡献度等信息。

### 5. templates/index.html

Flask应用的主页面模板，使用HTML、CSS和JavaScript构建。页面包括患者信息输入表单、预测结果展示区域以及SHAP解释图显示区域。

### 6. static/css/style.css

网页应用的样式文件，用于定义页面的布局、颜色、字体等样式。

### 7. static/js/main.js

网页应用的JavaScript文件，主要用于处理用户输入、发送预测请求、接收预测结果并更新页面显示。

### 8. requirements.txt

项目依赖的Python包列表，包括pandas、numpy、scikit-learn、xgboost、shap、flask等。通过 `pip install -r requirements.txt` 可以安装所有依赖项。

## 执行指南

1. **按顺序执行模块**：
   ```bash
   python data_generation.py    # 生成临床数据
   python model_training.py     # 训练和评估模型
   python model_explanation.py  # 生成SHAP可解释性分析
   python app.py                # 启动网页应用
   ```

2. **打开浏览器访问**：
   打开浏览器，访问`http://127.0.0.1:5000`，即可使用心肌梗死预后预测系统。

## 注意事项

- 确保在执行代码前已正确安装所有依赖项。
- 数据生成模块中的数据是模拟数据，实际应用中需要替换为真实的临床数据。
- 模型训练和评估过程中生成的图表和模型文件将保存在`plots`和`models`目录下。
- 网页应用中输入的患者数据需要符合实际临床范围，否则可能影响预测结果的准确性。

