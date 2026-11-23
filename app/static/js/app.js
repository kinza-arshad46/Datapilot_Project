/**
 * DataPilot AI — Frontend Application Logic
 * Professional SPA with full ML workflow
 */

const API = '/api';
let state = {
  token: localStorage.getItem('dp_token') || null,
  username: localStorage.getItem('dp_username') || null,
  userId: localStorage.getItem('dp_userId') || null,
  projectId: localStorage.getItem('dp_projectId') || null,
  profile: null,
  experiments: [],
  currentPage: 'home',
};

// ---------- Auth helpers ----------
function getToken() {
  // Always read fresh from localStorage (avoids stale memory state)
  const t = localStorage.getItem('dp_token');
  if (t) state.token = t;
  return state.token || t || null;
}

function authHeaders() {
  const token = getToken();
  if (!token) return {};
  return { Authorization: 'Bearer ' + token };
}

async function api(path, options = {}) {
  const isForm = options.body instanceof FormData;
  const headers = Object.assign({}, authHeaders(), options.headers || {});

  // Do NOT set Content-Type for FormData — browser must set multipart boundary
  if (!isForm && options.body && typeof options.body === 'object') {
    headers['Content-Type'] = 'application/json';
  }

  const opts = Object.assign({}, options, { headers: headers });
  if (opts.body && typeof opts.body === 'object' && !isForm) {
    opts.body = JSON.stringify(opts.body);
  }

  const res = await fetch(API + path, opts);

  if (res.status === 401) {
    state.token = null;
    localStorage.removeItem('dp_token');
    // Delay redirect so toast can be seen
    setTimeout(function () { window.location.href = '/login'; }, 1600);
    throw new Error('Session expired or not logged in. Please Sign In again.');
  }

  const data = await res.json().catch(function () { return {}; });
  if (!res.ok) {
    var detail = data.detail;
    var msg;
    if (typeof detail === 'string') msg = detail;
    else if (Array.isArray(detail)) msg = detail.map(function (d) { return d.msg || JSON.stringify(d); }).join(', ');
    else msg = data.message || 'Request failed';
    throw new Error(msg);
  }
  return data;
}

function logout(silent) {
  state.token = null;
  state.username = null;
  state.userId = null;
  state.projectId = null;
  localStorage.removeItem('dp_token');
  localStorage.removeItem('dp_username');
  localStorage.removeItem('dp_userId');
  localStorage.removeItem('dp_projectId');
  if (!silent) toast('Logged out', 'info');
  window.location.href = '/login';
}

// ---------- Toast ----------
function toast(msg, type = 'info') {
  const c = document.getElementById('toast-container');
  if (!c) return;
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span><span>${msg}</span>`;
  c.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ---------- Loading ----------
function showLoading(show = true) {
  const el = document.getElementById('loading-overlay');
  if (el) el.classList.toggle('hidden', !show);
}

// ---------- Navigation ----------
function navigate(page) {
  state.currentPage = page;
  document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
  const section = document.getElementById(`page-${page}`);
  if (section) section.classList.add('active');

  document.querySelectorAll('.nav-item').forEach(n => {
    n.classList.toggle('active', n.dataset.page === page);
  });

  const titles = {
    home: 'Home', dataset: 'Dataset', cleaning: 'Data Cleaning', eda: 'EDA Studio',
    'model-lab': 'Model Lab', comparison: 'Model Comparison', prediction: 'Prediction Studio',
    explainability: 'Explainability', dashboard: 'Dashboard', reports: 'Reports', settings: 'Settings',
  };
  const titleEl = document.getElementById('page-title');
  if (titleEl) titleEl.textContent = titles[page] || 'DataPilot AI';

  // Lazy load page data
  if (page === 'dashboard' && state.projectId) loadDashboard();
  if (page === 'comparison' && state.projectId) loadComparison();
  if (page === 'dataset' && state.projectId) loadProfile();
  if (page === 'cleaning' && state.projectId) loadCleaningSuggestions();
  if (page === 'eda' && state.projectId) loadEDAOverview();
  if (page === 'prediction' && state.projectId) loadPredictionSchema();
  if (page === 'model-lab' && state.projectId) { /* ready */ }
  if (page === 'reports' && state.projectId) { /* ready */ }

  // Mobile close
  document.querySelector('.sidebar')?.classList.remove('open');
}

// ---------- Login ----------
async function handleLogin(e) {
  e.preventDefault();
  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value;
  if (!username || !password) return toast('Enter username and password', 'error');

  showLoading(true);
  try {
    const data = await api('/auth/login', {
      method: 'POST',
      body: { username, password },
    });
    state.token = data.access_token;
    state.username = data.username;
    state.userId = data.user_id;
    localStorage.setItem('dp_token', data.access_token);
    localStorage.setItem('dp_username', data.username);
    localStorage.setItem('dp_userId', data.user_id);
    toast('Welcome back, ' + data.username, 'success');
    window.location.href = '/';
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    showLoading(false);
  }
}

async function handleRegister(e) {
  e.preventDefault();
  const username = document.getElementById('reg-username').value.trim();
  const email = document.getElementById('reg-email').value.trim();
  const password = document.getElementById('reg-password').value;
  if (!username || !password) return toast('Username and password required', 'error');

  showLoading(true);
  try {
    const data = await api('/auth/register', {
      method: 'POST',
      body: { username, email: email || null, password },
    });
    state.token = data.access_token;
    state.username = data.username;
    state.userId = data.user_id;
    localStorage.setItem('dp_token', data.access_token);
    localStorage.setItem('dp_username', data.username);
    localStorage.setItem('dp_userId', data.user_id);
    toast('Account created! Welcome, ' + data.username, 'success');
    window.location.href = '/';
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    showLoading(false);
  }
}

// ---------- Upload ----------
async function handleUpload(file) {
  if (!file) return;

  // Must be logged in
  const token = getToken();
  if (!token) {
    toast('Please Sign In first, then upload.', 'error');
    setTimeout(function () { window.location.href = '/login'; }, 1200);
    return;
  }

  // Basic client-side validation
  const allowed = ['.csv', '.xlsx', '.xls'];
  const name = (file.name || '').toLowerCase();
  if (!allowed.some(function (ext) { return name.endsWith(ext); })) {
    toast('Only CSV or Excel files are allowed', 'error');
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    toast('File is too large (max 50MB)', 'error');
    return;
  }

  const form = new FormData();
  form.append('file', file);
  form.append('project_name', (document.getElementById('project-name') && document.getElementById('project-name').value) || 'My Project');
  form.append('domain', (document.getElementById('domain-select') && document.getElementById('domain-select').value) || 'general');
  // Backup auth via form field (in case header is stripped)
  form.append('token', token);

  showLoading(true);
  try {
    const data = await api('/dataset/upload', { method: 'POST', body: form });
    state.projectId = data.project_id;
    state.profile = data.profile;
    localStorage.setItem('dp_projectId', String(data.project_id));
    toast('Dataset uploaded successfully!', 'success');
    renderProfile(data.profile);
    populateTargetSelect();
    navigate('dataset');
  } catch (err) {
    console.error('Upload error:', err);
    toast(err.message || 'Upload failed', 'error');
  } finally {
    showLoading(false);
  }
}

function setupUploadZone() {
  const zone = document.getElementById('upload-zone');
  const input = document.getElementById('file-input');
  if (!zone || !input) {
    console.warn('Upload zone or file input not found');
    return;
  }

  // File selected via click (overlay input covers whole zone)
  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    if (file) {
      handleUpload(file);
      input.value = ''; // allow selecting same file again
    }
  });

  // Drag & drop
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    zone.classList.remove('dragover');
  });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    zone.classList.remove('dragover');
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });
}

// ---------- Profile render ----------
function renderProfile(p) {
  if (!p) return;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('kpi-rows', p.rows?.toLocaleString() || '—');
  set('kpi-cols', p.columns || '—');
  set('kpi-missing', Object.values(p.missing_values || {}).reduce((a, b) => a + (b.count || 0), 0));
  set('kpi-dupes', p.duplicates || 0);
  set('kpi-health', (p.health_score || 0) + '%');
  set('profile-filename', p.filename || '—');
  set('profile-size', (p.memory_mb || 0) + ' MB');

  const ring = document.getElementById('health-ring');
  if (ring) ring.style.setProperty('--pct', p.health_score || 0);

  // Data types table
  const tbody = document.getElementById('dtypes-body');
  if (tbody && p.dtypes) {
    tbody.innerHTML = Object.entries(p.dtypes).map(([col, dt]) => {
      const miss = p.missing_values?.[col];
      return `<tr><td>${col}</td><td>${dt}</td><td>${miss ? miss.pct + '%' : '0%'}</td></tr>`;
    }).join('');
  }

  // Candidate targets
  const targets = document.getElementById('candidate-targets');
  if (targets && p.candidate_targets) {
    targets.innerHTML = p.candidate_targets.map(t =>
      `<div class="domain-card" style="cursor:default">
        <div class="domain-name">${t.column}</div>
        <div class="text-muted" style="font-size:0.75rem">${t.type} · ${t.nunique} unique</div>
      </div>`
    ).join('');
  }

  // Class dist pie if available
  if (p.class_distribution && Object.keys(p.class_distribution).length) {
    const labels = Object.keys(p.class_distribution);
    const values = Object.values(p.class_distribution).map(v => v * 100);
    Plotly.newPlot('class-dist-chart', [{
      type: 'pie', labels, values,
      marker: { colors: ['#1677FF', '#19C6FF', '#7257FF', '#3D8BFF', '#8B75FF'] },
      hole: 0.45,
    }], {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#E8F0FF', size: 11 },
      margin: { t: 30, b: 20, l: 20, r: 20 },
      showlegend: true,
    }, { responsive: true, displayModeBar: false });
  }
}

async function loadProfile() {
  if (!state.projectId) return;
  try {
    const p = await api(`/dataset/profile/${state.projectId}`);
    state.profile = p;
    renderProfile(p);
  } catch (e) { toast(e.message, 'error'); }
}

// ---------- Cleaning ----------
async function loadCleaningSuggestions() {
  if (!state.projectId) return;
  try {
    const s = await api(`/cleaning/suggestions/${state.projectId}`);
    const body = document.getElementById('missing-body');
    if (body) {
      body.innerHTML = Object.entries(s.missing || {}).map(([col, info]) =>
        `<tr>
          <td>${col}</td>
          <td>${info.pct}%</td>
          <td>
            <select class="form-control" style="padding:6px 10px;font-size:0.8rem" data-col="${col}">
              <option value="median" ${info.recommended==='median'?'selected':''}>Median</option>
              <option value="mean">Mean</option>
              <option value="most_frequent" ${info.recommended==='most_frequent'?'selected':''}>Most Frequent</option>
              <option value="drop">Drop rows</option>
            </select>
          </td>
        </tr>`
      ).join('') || '<tr><td colspan="3" class="text-muted">No missing values detected</td></tr>';
    }
    const dupEl = document.getElementById('dup-count');
    if (dupEl) dupEl.textContent = s.duplicates || 0;
  } catch (e) { /* ignore */ }
}

async function applyCleaning() {
  if (!state.projectId) return toast('Upload a dataset first', 'error');
  const strategies = {};
  document.querySelectorAll('#missing-body select').forEach(sel => {
    strategies[sel.dataset.col] = sel.value;
  });
  showLoading(true);
  try {
    const data = await api('/cleaning/apply', {
      method: 'POST',
      body: {
        project_id: parseInt(state.projectId),
        missing_strategies: strategies,
        drop_duplicates: document.getElementById('drop-dupes')?.checked ?? true,
        outlier_action: document.getElementById('outlier-action')?.value || 'none',
      },
    });
    state.profile = data.profile;
    toast(`Cleaning applied: ${data.original_shape} → ${data.new_shape}`, 'success');
    renderProfile(data.profile);
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    showLoading(false);
  }
}

// ---------- EDA ----------
async function loadEDAOverview() {
  if (!state.projectId) return;
  try {
    const data = await api(`/eda/overview/${state.projectId}`);
    const numSelect = document.getElementById('eda-num-col');
    const catSelect = document.getElementById('eda-cat-col');
    if (numSelect) {
      numSelect.innerHTML = (data.numerical || []).map(c => `<option value="${c}">${c}</option>`).join('');
      if (data.numerical?.length) loadNumDist(data.numerical[0]);
    }
    if (catSelect) {
      catSelect.innerHTML = (data.categorical || []).map(c => `<option value="${c}">${c}</option>`).join('');
      if (data.categorical?.length) loadCatFreq(data.categorical[0]);
    }
    // Correlation
    try {
      const corr = await api(`/eda/correlation/${state.projectId}`);
      Plotly.newPlot('corr-chart', corr.chart.data, corr.chart.layout, { responsive: true, displayModeBar: false });
    } catch (_) {}
  } catch (e) { toast(e.message, 'error'); }
}

async function loadNumDist(col) {
  if (!col || !state.projectId) return;
  try {
    const data = await api(`/eda/distribution/${state.projectId}?column=${encodeURIComponent(col)}`);
    Plotly.newPlot('hist-chart', data.histogram.data, data.histogram.layout, { responsive: true, displayModeBar: false });
    Plotly.newPlot('box-chart', data.boxplot.data, data.boxplot.layout, { responsive: true, displayModeBar: false });
  } catch (e) { /* */ }
}

async function loadCatFreq(col) {
  if (!col || !state.projectId) return;
  try {
    const data = await api(`/eda/categorical/${state.projectId}?column=${encodeURIComponent(col)}`);
    Plotly.newPlot('cat-chart', data.chart.data, data.chart.layout, { responsive: true, displayModeBar: false });
  } catch (e) { /* */ }
}

// ---------- Model Lab ----------
async function trainModels() {
  if (!state.projectId) return toast('Upload a dataset first', 'error');
  const target = document.getElementById('target-col')?.value;
  const problemType = document.getElementById('problem-type')?.value || 'classification';
  if (problemType !== 'clustering' && !target) return toast('Select a target column', 'error');

  const algos = [];
  document.querySelectorAll('.algo-check:checked').forEach(c => algos.push(c.value));

  showLoading(true);
  const progress = document.getElementById('train-progress');
  if (progress) progress.classList.remove('hidden');

  try {
    const data = await api('/models/train', {
      method: 'POST',
      body: {
        project_id: parseInt(state.projectId),
        target: target || '',
        problem_type: problemType,
        algorithms: algos,
        test_size: parseFloat(document.getElementById('test-size')?.value || 0.2),
        scaler: document.getElementById('scaler')?.value || 'standard',
        n_clusters: parseInt(document.getElementById('n-clusters')?.value || 3),
      },
    });
    state.experiments = data.results || [];
    toast(`Trained ${data.results.filter(r => r.status === 'success').length} models`, 'success');
    renderTrainResults(data.results);
    if (progress) {
      progress.querySelectorAll('.progress-step').forEach(s => s.classList.add('done'));
    }
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    showLoading(false);
  }
}

function renderTrainResults(results) {
  const el = document.getElementById('train-results');
  if (!el) return;
  el.innerHTML = results.map(r => {
    if (r.status !== 'success') {
      return `<div class="model-card"><strong>${r.algorithm}</strong> <span class="text-muted">Failed: ${r.error}</span></div>`;
    }
    const m = r.metrics || {};
    const score = m.accuracy ?? m.r2 ?? m.silhouette ?? '—';
    return `<div class="model-card">
      <div class="flex-between">
        <strong>${r.algorithm}</strong>
        <span class="badge badge-success">${typeof score === 'number' ? score.toFixed(4) : score}</span>
      </div>
      <div class="text-muted mt-1" style="font-size:0.8rem">
        ${m.accuracy != null ? `Acc: ${m.accuracy.toFixed(3)} · F1: ${(m.f1||0).toFixed(3)} · AUC: ${(m.roc_auc||0).toFixed(3)}` : ''}
        ${m.r2 != null ? `R²: ${m.r2.toFixed(3)} · RMSE: ${(m.rmse||0).toFixed(3)}` : ''}
        ${m.silhouette != null ? `Silhouette: ${m.silhouette.toFixed(3)} · Clusters: ${m.n_clusters}` : ''}
      </div>
    </div>`;
  }).join('');
}

// ---------- Comparison ----------
async function loadComparison() {
  if (!state.projectId) return;
  try {
    const exps = await api(`/models/experiments/${state.projectId}`);
    state.experiments = exps;
    const tbody = document.getElementById('comparison-body');
    if (!tbody) return;
    if (!exps.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="text-muted text-center">No models trained yet. Go to Model Lab.</td></tr>';
      return;
    }
    // Find best by accuracy or r2
    let bestIdx = 0, bestScore = -Infinity;
    exps.forEach((e, i) => {
      const s = e.metrics?.accuracy ?? e.metrics?.r2 ?? e.metrics?.silhouette ?? 0;
      if (s > bestScore) { bestScore = s; bestIdx = i; }
    });
    tbody.innerHTML = exps.map((e, i) => {
      const m = e.metrics || {};
      const isBest = i === bestIdx;
      return `<tr class="${isBest ? 'best-row' : ''}">
        <td>${e.algorithm} ${isBest ? '<span class="badge badge-best">Best</span>' : ''}</td>
        <td>${m.accuracy != null ? m.accuracy.toFixed(4) : '—'}</td>
        <td>${m.precision != null ? m.precision.toFixed(4) : '—'}</td>
        <td>${m.recall != null ? m.recall.toFixed(4) : '—'}</td>
        <td>${m.f1 != null ? m.f1.toFixed(4) : '—'}</td>
        <td>${m.roc_auc != null ? m.roc_auc.toFixed(4) : (m.r2 != null ? m.r2.toFixed(4) : '—')}</td>
        <td>${m.rmse != null ? m.rmse.toFixed(4) : '—'}</td>
      </tr>`;
    }).join('');
  } catch (e) { toast(e.message, 'error'); }
}

// ---------- Prediction ----------
async function loadPredictionSchema() {
  if (!state.projectId) return;
  try {
    const exps = await api(`/models/experiments/${state.projectId}`);
    const sel = document.getElementById('pred-model-select');
    if (!sel) return;
    sel.innerHTML = exps.map(e => `<option value="${e.id}">${e.algorithm} (#${e.id})</option>`).join('');
    if (exps.length) loadSchemaForExp(exps[0].id);
    sel.onchange = () => loadSchemaForExp(sel.value);
  } catch (e) { /* */ }
}

async function loadSchemaForExp(expId) {
  try {
    const schema = await api(`/prediction/schema/${expId}`);
    const form = document.getElementById('pred-form-fields');
    if (!form) return;
    form.innerHTML = (schema.schema || []).map(f =>
      `<div class="form-group">
        <label class="form-label">${f.name} ${f.required ? '*' : ''}</label>
        <input class="form-control pred-input" data-name="${f.name}" type="${f.type === 'number' ? 'number' : 'text'}"
               step="any" placeholder="Enter ${f.name}" ${f.required ? 'required' : ''}>
      </div>`
    ).join('') || '<p class="text-muted">No input schema available</p>';
  } catch (e) { toast(e.message, 'error'); }
}

async function runPrediction() {
  const expId = document.getElementById('pred-model-select')?.value;
  if (!expId) return toast('Select a model', 'error');
  const features = {};
  document.querySelectorAll('.pred-input').forEach(inp => {
    const v = inp.value;
    features[inp.dataset.name] = inp.type === 'number' ? parseFloat(v) : v;
  });
  showLoading(true);
  try {
    const data = await api('/prediction/predict', {
      method: 'POST',
      body: { experiment_id: parseInt(expId), features },
    });
    const res = data.result || {};
    const panel = document.getElementById('pred-result-panel');
    if (panel) {
      panel.classList.remove('hidden');
      document.getElementById('pred-label').textContent =
        res.prediction_label ?? res.prediction ?? '—';
      if (res.probabilities) {
        const probaHtml = Object.entries(res.probabilities)
          .map(([k, v]) => `<div class="flex-between" style="margin:4px 0"><span>${k}</span><span>${(v*100).toFixed(1)}%</span></div>
            <div class="progress-bar"><div class="progress-fill" style="width:${v*100}%"></div></div>`)
          .join('');
        document.getElementById('pred-proba').innerHTML = probaHtml;
      }
    }
    // Importance bars
    if (data.feature_importance?.length) {
      const maxImp = Math.max(...data.feature_importance.map(f => Math.abs(f.importance))) || 1;
      document.getElementById('pred-importance').innerHTML = data.feature_importance.slice(0, 8).map(f =>
        `<div style="margin-bottom:8px">
          <div class="flex-between" style="font-size:0.8rem"><span>${f.feature}</span><span>${f.importance}</span></div>
          <div class="progress-bar"><div class="progress-fill" style="width:${Math.abs(f.importance)/maxImp*100}%;background:linear-gradient(90deg,#7257FF,#19C6FF)"></div></div>
        </div>`
      ).join('');
    }
    toast('Prediction complete', 'success');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    showLoading(false);
  }
}

// ---------- Dashboard ----------
async function loadDashboard() {
  if (!state.projectId) return;
  try {
    const d = await api(`/dashboard/${state.projectId}`);
    const k = d.kpis || {};
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('dash-rows', (k.rows || 0).toLocaleString());
    set('dash-features', k.features || 0);
    set('dash-models', k.model_count || 0);
    set('dash-best', k.best_metric != null ? k.best_metric.toFixed(3) : '—');
    set('dash-preds', k.prediction_count || 0);
    set('dash-health', (k.health_score || 0) + '%');

    // Model performance bars
    if (d.model_performance?.length) {
      Plotly.newPlot('dash-model-perf', [{
        type: 'bar',
        x: d.model_performance.map(m => m.algorithm),
        y: d.model_performance.map(m => m.score),
        marker: { color: ['#1677FF', '#19C6FF', '#7257FF', '#3D8BFF', '#8B75FF'] },
      }], {
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(7,26,61,0.4)',
        font: { color: '#E8F0FF', size: 11 },
        margin: { t: 30, b: 60, l: 40, r: 20 },
        yaxis: { gridcolor: 'rgba(22,119,255,0.1)' },
        xaxis: { tickangle: -20 },
      }, { responsive: true, displayModeBar: false });
    }

    // Class dist
    if (d.class_distribution && Object.keys(d.class_distribution).length) {
      Plotly.newPlot('dash-class-dist', [{
        type: 'pie',
        labels: Object.keys(d.class_distribution),
        values: Object.values(d.class_distribution).map(v => typeof v === 'number' ? v * 100 : v),
        marker: { colors: ['#1677FF', '#19C6FF', '#7257FF'] },
        hole: 0.5,
      }], {
        paper_bgcolor: 'rgba(0,0,0,0)', font: { color: '#E8F0FF', size: 11 },
        margin: { t: 20, b: 20, l: 20, r: 20 }, showlegend: true,
      }, { responsive: true, displayModeBar: false });
    }
  } catch (e) { toast(e.message, 'error'); }
}

// ---------- Reports ----------
async function generateReport() {
  if (!state.projectId) return toast('Upload a dataset first', 'error');
  showLoading(true);
  try {
    const data = await api(`/reports/generate/${state.projectId}`, { method: 'POST' });
    toast('Report generated!', 'success');
    const link = document.getElementById('report-download-link');
    if (link) {
      link.href = data.path;
      link.classList.remove('hidden');
      link.textContent = 'Download PDF — ' + data.filename;
    }
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    showLoading(false);
  }
}

// ---------- Populate target select from profile ----------
function populateTargetSelect() {
  const sel = document.getElementById('target-col');
  if (!sel || !state.profile) return;
  const cols = Object.keys(state.profile.dtypes || {});
  sel.innerHTML = cols.map(c => `<option value="${c}">${c}</option>`).join('');
}

// ---------- Init ----------
document.addEventListener('DOMContentLoaded', () => {
  // Login page
  const loginForm = document.getElementById('login-form');
  if (loginForm) {
    loginForm.addEventListener('submit', handleLogin);
    document.getElementById('register-form')?.addEventListener('submit', handleRegister);
    document.getElementById('show-register')?.addEventListener('click', e => {
      e.preventDefault();
      document.getElementById('login-panel').classList.add('hidden');
      document.getElementById('register-panel').classList.remove('hidden');
    });
    document.getElementById('show-login')?.addEventListener('click', e => {
      e.preventDefault();
      document.getElementById('register-panel').classList.add('hidden');
      document.getElementById('login-panel').classList.remove('hidden');
    });
    return;
  }

  // Main app auth check
  if (!state.token) {
    window.location.href = '/login';
    return;
  }

  // Set username
  const userEl = document.getElementById('username-display');
  if (userEl) userEl.textContent = state.username || 'User';
  const avatar = document.getElementById('user-avatar');
  if (avatar) avatar.textContent = (state.username || 'U')[0].toUpperCase();

  // Nav
  document.querySelectorAll('.nav-item[data-page]').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      navigate(item.dataset.page);
    });
  });

  // Mobile menu
  document.getElementById('mobile-menu-btn')?.addEventListener('click', () => {
    document.querySelector('.sidebar')?.classList.toggle('open');
  });

  setupUploadZone();

  // Buttons
  document.getElementById('btn-apply-cleaning')?.addEventListener('click', applyCleaning);
  document.getElementById('btn-train')?.addEventListener('click', trainModels);
  document.getElementById('btn-predict')?.addEventListener('click', runPrediction);
  document.getElementById('btn-generate-report')?.addEventListener('click', generateReport);
  document.getElementById('btn-logout')?.addEventListener('click', () => logout());

  // EDA selects
  document.getElementById('eda-num-col')?.addEventListener('change', e => loadNumDist(e.target.value));
  document.getElementById('eda-cat-col')?.addEventListener('change', e => loadCatFreq(e.target.value));

  // Problem type toggle
  document.getElementById('problem-type')?.addEventListener('change', e => {
    const isCluster = e.target.value === 'clustering';
    document.getElementById('target-group')?.classList.toggle('hidden', isCluster);
    document.getElementById('cluster-group')?.classList.toggle('hidden', !isCluster);
  });

  // Start on home
  navigate('home');

  // If project exists, load profile
  if (state.projectId) {
    loadProfile().then(() => populateTargetSelect());
  }
});
