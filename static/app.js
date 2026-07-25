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

function setExample() {
  noteEl.value = EXAMPLES[exampleEl.value] || "";
}

function showError(message) {
  errorEl.hidden = !message;
  errorEl.textContent = message || "";
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.model_loaded) {
      statusEl.textContent = "Model ready";
      predictBtn.disabled = false;
    } else {
      statusEl.textContent = "Model not loaded";
      predictBtn.disabled = true;
    }
  } catch {
    statusEl.textContent = "API unreachable";
    predictBtn.disabled = true;
  }
}

async function predict() {
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

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting…";

  try {
    const res = await fetch("/triage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, confirm_no_phi: true }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Prediction failed");
    }

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
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict specialty";
  }
}

exampleEl.addEventListener("change", setExample);
predictBtn.addEventListener("click", predict);
setExample();
predictBtn.disabled = true;
checkHealth();
