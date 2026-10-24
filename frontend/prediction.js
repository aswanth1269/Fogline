/* Shared forecasting view; uses the app's API, escaping and feedback primitives. */
let analysisPending = false;
let renderedPredictionModel = null;
function investigationID() {
  const input = $('#investigation-id');
  const value = input.value.trim();
  const valid = /^[A-Za-z0-9_.-]{1,64}$/.test(value);
  input.setAttribute('aria-invalid', String(!valid));
  if (!valid) {
    input.focus();
    throw new Error('Use 1–64 letters, digits, dots, dashes or underscores for the investigation ID.');
  }
  try { localStorage.setItem('fogline-investigation', value); } catch (_) { /* In-memory input still works. */ }
  return value;
}
function initializeInvestigation() {
  let saved;
  try { saved = localStorage.getItem('fogline-investigation'); } catch (_) { /* Storage optional. */ }
  $('#investigation-id').value = saved || `case-${crypto.randomUUID().slice(0, 8)}`;
  $('#investigation-id').addEventListener('change', () => {
    try { investigationID(); $('#analysis-error').textContent = ''; }
    catch (error) { $('#analysis-error').textContent = error.message; }
  });
  $('#prediction-train-form').addEventListener('submit', trainNextMove);
}
function newInvestigation() {
  if (analysisPending) return;
  $('#investigation-id').value = `case-${crypto.randomUUID().slice(0, 8)}`;
  investigationID();
  $('#sequence-progress').textContent = 'New investigation · 0 / 3 observations';
  $('#analysis-error').textContent = '';
  $('#result').innerHTML = '<div class="empty"><h2>New investigation started</h2><p>Add three related observations to produce a forecast. Previous events remain in event history.</p></div>';
  toast('New investigation started; earlier events are preserved');
}
async function analyzeBusy(button, fn, errorTarget) {
  if (analysisPending) return;
  analysisPending = true;
  const controls = [$('#run'), $('#demo-next'), $('#new-investigation'), $('#investigation-id'), $('#custom-form button[type=submit]')];
  await busy(button, async () => {
    controls.forEach(control => { if (control !== button) control.disabled = true; });
    try { await showAnalysis(fn); }
    finally { controls.forEach(control => { if (control !== button) control.disabled = false; }); }
  }, errorTarget);
  analysisPending = false;
}
function runDemoStep() {
  analyzeBusy($('#demo-next'), () => post('/api/prediction/demo-step', {investigation_id: investigationID()}), $('#analysis-error'));
}
function forecastHTML(event) {
  const forecast = event.next_move;
  if (!forecast) return '<div class="notice"><strong>No forecast recorded</strong><span>This event predates sequence forecasting. Add new related observations in the traffic lab.</span></div>';
  const demo = forecast.training_source === 'synthetic';
  const candidates = forecast.candidates || [];
  const canForecast = forecast.status === 'forecast';
  const heading = canForecast ? 'Most likely next observed behavior' : forecast.status === 'uncertain' ? 'Next move is inconclusive' : 'Next-move forecast pending';
  const evidence = (forecast.history || []).map((item, index) => `<li>${item.event_id ? `EV-${String(item.event_id).padStart(4, '0')}` : 'Current observation'}<small>${new Date(item.timestamp).toLocaleTimeString()}</small></li>`).join('');
  return `<section class="forecast-box" aria-label="Next-move prediction"><div class="eyebrow">${demo ? 'DEMONSTRATION FORECAST · SYNTHETIC TRAINING' : 'NEXT-MOVE PREDICTION'}</div><h2>${heading}</h2>${canForecast ? `<h3 class="next-behavior">${escapeHTML(candidates[0].label)}</h3>` : ''}<p>${escapeHTML(forecast.message)}</p>${forecast.history_count !== undefined ? `<p class="hint">Investigation ${escapeHTML(forecast.investigation_id)} · ${forecast.history_count} / 3 observations · Model v${forecast.model_version}</p>` : ''}${evidence ? `<ol class="observation-trail" aria-label="Observations used for this forecast">${evidence}</ol>` : ''}${candidates.length ? `<div class="candidate-list">${candidates.map(candidate => `<div>${meter(candidate.label, candidate.score * 100)}</div>`).join('')}</div><p class="hint">Uncalibrated model scores, not verified probabilities. Only the top three are shown. Continuing the same behavior is a valid next outcome.</p>${canForecast ? `<div class="next-check"><strong>Evidence to check next</strong><p>${escapeHTML(candidates[0].recommendation)}</p></div>` : ''}` : ''}${forecast.training_source ? `<p class="forecast-limit">${demo ? 'Trained on generated scenarios. This demonstrates the workflow; it is not a validated real-intrusion forecast.' : 'Trained on uploaded sequences; their provenance and real-world generalization are unverified.'} ${escapeHTML(forecast.limitation || '')}</p>` : ''}</section>`;
}
function renderPredictionModel(model) {
  if (!model) { $('#prediction-model-status').textContent = 'Restart the backend to load sequence forecasting.'; return; }
  const signature = `${model.version}:${model.dataset_sha256}`;
  if (signature === renderedPredictionModel) return;
  renderedPredictionModel = signature;
  const m = model.metrics;
  $('#prediction-model-status').innerHTML = `<div class="notice"><strong>v${model.version} · ${model.source === 'synthetic' ? 'Synthetic demonstration sequences' : 'Uploaded sequences · provenance unverified'}</strong><span>${escapeHTML(model.method)}. ${escapeHTML(model.target)}</span></div><div class="forecast-metrics"><div><strong>${(m.accuracy * 100).toFixed(1)}%</strong><span>Next-event accuracy</span></div><div><strong>${(m.macro_f1 * 100).toFixed(1)}%</strong><span>Macro F1</span></div><div><strong>${(m.majority_accuracy * 100).toFixed(1)}%</strong><span>Training-majority baseline</span></div></div><p class="hint">${m.train_windows} earlier training windows / ${m.test_windows} later holdout windows · ${m.train_episodes} training episodes / ${m.test_episodes} holdout episodes · ${m.purged_episodes} crossing episodes excluded. Scores describe this dataset only.</p><details><summary>Per-behavior holdout results</summary><div class="table-scroll"><table><thead><tr><th>Behavior</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>${Object.entries(m.per_class).filter(([key]) => !['accuracy', 'macro avg', 'weighted avg', 'micro avg'].includes(key)).map(([key, value]) => `<tr><td>${escapeHTML(key)}</td><td>${(value.precision * 100).toFixed(1)}%</td><td>${(value.recall * 100).toFixed(1)}%</td><td>${(value['f1-score'] * 100).toFixed(1)}%</td><td>${value.support}</td></tr>`).join('')}</tbody></table></div><p class="hint">Last training target: ${escapeHTML(m.train_end)}. First holdout target: ${escapeHTML(m.test_start)}.${m.unseen_holdout_classes.length ? ` Unseen holdout behaviors: ${escapeHTML(m.unseen_holdout_classes.join(', '))}.` : ''}</p></details>`;
}
function trainNextMove(event) {
  event.preventDefault();
  const input = $('#sequence-dataset');
  const file = input.files[0];
  if (!file || !file.name.toLowerCase().endsWith('.csv') || file.size > 10 * 1024 * 1024) {
    $('#prediction-train-error').textContent = 'Choose a labeled sequence CSV no larger than 10 MB.';
    input.setAttribute('aria-invalid', 'true');
    input.focus();
    return;
  }
  input.setAttribute('aria-invalid', 'false');
  busy(event.submitter, async () => {
    $('#prediction-train-status').textContent = 'Training and evaluating on later, separate episodes…';
    const form = new FormData();
    form.append('file', file);
    try {
      const response = await api('/api/prediction/train', {method: 'POST', body: form});
      renderPredictionModel(response.next_move_model);
      $('#prediction-train-status').textContent = `Next-move model v${response.next_move_model.version} saved. New observations use this version; historical forecasts are preserved.`;
      toast('Next-move model trained and saved');
    } catch (error) {
      $('#prediction-train-status').textContent = 'Training failed. The previous forecasting model remains active.';
      throw error;
    }
  }, $('#prediction-train-error'));
}
