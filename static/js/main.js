// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultsDiv = document.getElementById('results');

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        // 收集表单数据
        const formData = new FormData(form);
        const data = {};

        // 验证表单是否有效
        let isValid = true;

        for (const [key, value] of formData.entries()) {
            const input = document.getElementById(key);
            data[key] = parseFloat(value);

            // 对于数值输入进行验证
            if (input.type === 'number') {
                const min = parseFloat(input.min);
                const max = parseFloat(input.max);

                if (isNaN(data[key]) || data[key] < min || data[key] > max) {
                    input.classList.add('is-invalid');
                    isValid = false;
                } else {
                    input.classList.remove('is-invalid');
                }
            }
        }

        if (!isValid) {
            alert('请修正表单中的错误');
            return;
        }

        // 发送预测请求
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(result => {
            // 显示结果区域
            resultsDiv.style.display = 'block';

            // 设置预测标签和概率
            const labelSpan = document.getElementById('prediction-label');
            labelSpan.textContent = result.label_text;
            labelSpan.className = 'badge bg-' + (result.label === 1 ? 'danger' : 'success');

            // 设置预测概率
            document.getElementById('prediction-probability').textContent = (result.probability * 100).toFixed(2) + '%';

            // 设置进度条
            const probabilityBar = document.getElementById('probability-bar');
            probabilityBar.style.width = (result.probability * 100) + '%';
            probabilityBar.className = 'progress-bar bg-' + (result.label === 1 ? 'danger' : 'success');

            // 设置风险等级
            const riskSpan = document.getElementById('risk-level');
            riskSpan.textContent = result.risk_level.level;
            riskSpan.className = 'badge bg-' + result.risk_level.color;

            // 设置SHAP图像
            document.getElementById('shap-image').src = 'data:image/png;base64,' + result.shap_image;

            // 设置特征贡献度
            const contributionsDiv = document.getElementById('feature-contributions');
            contributionsDiv.innerHTML = '';

            // 只显示前5个贡献最大的特征
            const topContributions = result.feature_contributions.slice(0, 5);

            for (const [feature, contribution] of topContributions) {
                const featureItem = document.createElement('div');
                featureItem.className = 'feature-item';

                const featureName = document.createElement('div');
                featureName.className = 'd-flex justify-content-between';

                const nameSpan = document.createElement('span');
                nameSpan.textContent = feature.replace('_', ' ');

                const valueSpan = document.createElement('span');
                valueSpan.className = 'feature-value';
                valueSpan.textContent = contribution.toFixed(1) + '%';

                featureName.appendChild(nameSpan);
                featureName.appendChild(valueSpan);

                const progressDiv = document.createElement('div');
                progressDiv.className = 'progress';

                const progressBar = document.createElement('div');
                progressBar.className = 'progress-bar contribution-bar bg-primary';
                progressBar.style.width = contribution + '%';

                progressDiv.appendChild(progressBar);

                featureItem.appendChild(featureName);
                featureItem.appendChild(progressDiv);

                contributionsDiv.appendChild(featureItem);
            }

            // 滚动到结果区域
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            console.error('预测出错:', error);
            alert('预测出错，请检查日志');
        });
    });
});
