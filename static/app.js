/** Client-side TextCNN inference from pretrained static/model.json (no server). */

const EXAMPLES = {
  Kardiologie:
    "Patient berichtet ueber belastungsabhaengige Brustschmerzen und Dyspnoe. Troponin leicht erhoeht.",
  Neurologie:
    "Akute Halbseitenschwaeche rechts und Sprachstoerung seit heute morgen.",
  Orthopaedie:
    "Chronische Rueckenschmerzen lumbal mit Ausstrahlung ins Bein.",
  "Innere Medizin":
    "Seit Wochen unklare Gewichtsabnahme, Nachtschweiss und Muedigkeit.",
  Notfallmedizin:
    "Akute Atemnot und Zyanose, Verdacht auf Lungenembolie. Vitalparameter instabil.",
};

const noteEl = document.getElementById("note");
const exampleEl = document.getElementById("example");
const predictBtn = document.getElementById("predict");
const noPhiEl = document.getElementById("no-phi");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const resultEmpty = document.getElementById("result-empty");
const resultEl = document.getElementById("result");
const predictionEl = document.getElementById("prediction");
const rankingEl = document.getElementById("ranking");
const disclaimerEl = document.getElementById("disclaimer");

let model = null;

function setExample() {
  noteEl.value = EXAMPLES[exampleEl.value] || "";
}

function showError(message) {
  errorEl.hidden = !message;
  errorEl.textContent = message || "";
}

function tokenize(text) {
  return text.toLowerCase().replaceAll(",", " ").replaceAll(".", " ").split(/\s+/).filter(Boolean);
}

function encode(tokens, stoi, unkId, padId, maxLen) {
  const ids = tokens.slice(0, maxLen).map((t) => (t in stoi ? stoi[t] : unkId));
  while (ids.length < maxLen) ids.push(padId);
  return ids;
}

function relu(x) {
  return x > 0 ? x : 0;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/** Conv1d over (embed_dim, seq_len) with weight (out_channels, in_channels, k). */
function conv1dMax(channels, weight, bias) {
  const outChannels = weight.length;
  const inChannels = weight[0].length;
  const kernel = weight[0][0].length;
  const seqLen = channels[0].length;
  const outLen = seqLen - kernel + 1;
  const pooled = new Array(outChannels);

  for (let oc = 0; oc < outChannels; oc++) {
    let best = -Infinity;
    for (let t = 0; t < outLen; t++) {
      let sum = bias[oc];
      for (let ic = 0; ic < inChannels; ic++) {
        for (let k = 0; k < kernel; k++) {
          sum += weight[oc][ic][k] * channels[ic][t + k];
        }
      }
      best = Math.max(best, relu(sum));
    }
    pooled[oc] = best;
  }
  return pooled;
}

function predictLocal(text) {
  const ids = encode(
    tokenize(text),
    model.stoi,
    model.unk_id,
    model.pad_id,
    model.max_len
  );

  // embedding: (embed_dim, seq)
  const channels = Array.from({ length: model.embed_dim }, () => new Array(model.max_len));
  for (let t = 0; t < model.max_len; t++) {
    const row = model.embedding[ids[t]];
    for (let d = 0; d < model.embed_dim; d++) channels[d][t] = row[d];
  }

  const features = [];
  for (const conv of model.convs) {
    features.push(...conv1dMax(channels, conv.weight, conv.bias));
  }

  const logits = model.fc.bias.map((b, i) => {
    let sum = b;
    for (let j = 0; j < features.length; j++) sum += model.fc.weight[i][j] * features[j];
    return sum;
  });

  const probs = softmax(logits);
  const ranking = model.specialties
    .map((specialty, i) => ({ specialty, probability: probs[i] }))
    .sort((a, b) => b.probability - a.probability);

  return {
    predicted_specialty: ranking[0].specialty,
    confidence: ranking[0].probability,
    ranking,
    disclaimer: model.disclaimer,
  };
}

function renderResult(data) {
  resultEmpty.hidden = true;
  resultEl.hidden = false;
  predictionEl.textContent = `${data.predicted_specialty}  (${(data.confidence * 100).toFixed(1)}%)`;
  disclaimerEl.textContent = data.disclaimer;

  rankingEl.innerHTML = "";
  for (const row of data.ranking) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = row.specialty;
    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = `${(row.probability * 100).toFixed(1)}%`;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    bar.appendChild(fill);
    li.append(name, pct, bar);
    rankingEl.appendChild(li);
    requestAnimationFrame(() => {
      fill.style.width = `${Math.max(row.probability * 100, 2)}%`;
    });
  }
}

async function loadModel() {
  try {
    const res = await fetch(new URL("model.json", window.location.href));
    if (!res.ok) throw new Error(`Could not load model.json (${res.status})`);
    model = await res.json();
    statusEl.textContent = "Pretrained model ready (browser inference)";
    predictBtn.disabled = false;
  } catch (err) {
    statusEl.textContent = "Model not loaded";
    showError(err.message || String(err));
  }
}

function predict() {
  showError("");
  const text = noteEl.value.trim();
  if (text.length < 10) {
    showError("Please enter a longer clinical note.");
    return;
  }
  if (!noPhiEl.checked) {
    showError("Confirm that the note contains no real patient data.");
    return;
  }
  if (!model) {
    showError("Model is not loaded yet.");
    return;
  }

  try {
    renderResult(predictLocal(text));
  } catch (err) {
    showError(err.message || String(err));
  }
}

exampleEl.addEventListener("change", setExample);
predictBtn.addEventListener("click", predict);
setExample();
predictBtn.disabled = true;
loadModel();
