function fillExample(value) {
  const smilesInput = document.getElementById('smilesInput');
  if (smilesInput) smilesInput.value = value;
}

function toHtmlTable(rows) {
  if (!rows || rows.length === 0) return '<p class="text-secondary">无</p>';
  const header = '<table class="table table-sm table-striped"><thead><tr><th>特征</th><th>SHAP值</th><th>方向</th></tr></thead><tbody>';
  const body = rows.map(row => `<tr><td>${row.feature}</td><td>${Number(row.shap_value).toFixed(4)}</td><td>${row.direction}</td></tr>`).join('');
  return header + body + '</tbody></table>';
}

async function predictSmiles() {
  const smiles = document.getElementById('smilesInput').value;
  const status = document.getElementById('predictStatus');
  const image = document.getElementById('moleculeImage');
  const placeholder = document.getElementById('moleculePlaceholder');
  status.textContent = '正在预测，请稍候...';
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({smiles})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '预测失败');

    image.src = data.molecule_image;
    image.classList.remove('d-none');
    placeholder.classList.add('d-none');
    document.getElementById('predictSummary').innerHTML = `
      <h5 class="fw-bold mb-3">预测结果</h5>
      <p><strong>标准化 SMILES：</strong><code>${data.canonical_smiles}</code></p>
      <p><strong>集成模型预测脂溶性：</strong>${Number(data.ensemble_prediction).toFixed(4)}</p>
      <p><strong>解释模型（CatBoost）预测：</strong>${Number(data.catboost_prediction).toFixed(4)}</p>
      <p><strong>解读：</strong>${data.interpretation}</p>
    `;
    document.getElementById('positiveTable').innerHTML = toHtmlTable(data.top_positive_features);
    document.getElementById('negativeTable').innerHTML = toHtmlTable(data.top_negative_features);
    status.textContent = '预测完成。';
  } catch (error) {
    image.classList.add('d-none');
    image.removeAttribute('src');
    placeholder.classList.remove('d-none');
    status.textContent = '错误：' + error.message;
  }
}

async function geminiExplain() {
  const smiles = document.getElementById('smilesInput').value;
  const focus = document.getElementById('geminiFocus').value;
  const status = document.getElementById('predictStatus');
  const geminiBox = document.getElementById('geminiOutput');
  status.textContent = 'Gemini 正在生成解释，请稍候...';
  geminiBox.innerHTML = 'Gemini 正在生成解释，请稍候...';
  try {
    const response = await fetch('/api/gemini-explain', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({smiles, focus})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Gemini 解读失败');
    geminiBox.innerHTML = data.analysis_markdown.replace(/\n/g, '<br>');
    status.textContent = 'Gemini 解读完成。';
  } catch (error) {
    geminiBox.innerHTML = 'Gemini 调用失败：' + error.message;
    status.textContent = 'Gemini 调用失败。';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const predictBtn = document.getElementById('predictBtn');
  const geminiBtn = document.getElementById('geminiBtn');
  const exampleButtons = document.querySelectorAll('.example-btn');

  if (predictBtn) predictBtn.addEventListener('click', predictSmiles);
  if (geminiBtn) geminiBtn.addEventListener('click', geminiExplain);
  exampleButtons.forEach((button) => {
    button.addEventListener('click', () => fillExample(button.dataset.example));
  });
});
