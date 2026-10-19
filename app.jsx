import { useCallback, useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const initialForm = {
  Year: "2024",
  dmin: "0.0",
  longitude: "0.0",
  nst: "10",
  latitude: "0.0",
  depth: "10",
  gap: "90",
  sig: "100"
};

const fields = [
  { name: "Year", label: "Year", hint: "Event year", step: "1", min: "1800", max: "2200" },
  { name: "dmin", label: "Minimum distance (dmin)", hint: "Use the same units as training data", step: "any", min: "0" },
  { name: "longitude", label: "Longitude", hint: "−180 to 180 degrees", step: "any", min: "-180", max: "180" },
  { name: "nst", label: "Number of stations (nst)", hint: "Non-negative integer", step: "1", min: "0" },
  { name: "latitude", label: "Latitude", hint: "−90 to 90 degrees", step: "any", min: "-90", max: "90" },
  { name: "depth", label: "Depth", hint: "Use the same units as training data", step: "any", min: "0" },
  { name: "gap", label: "Gap", hint: "Use the same units as training data", step: "any", min: "0" },
  { name: "sig", label: "Significance (sig)", hint: "Non-negative numeric value", step: "any", min: "0" }
];

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [historyError, setHistoryError] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("predict");
  const [apiStatus, setApiStatus] = useState("checking");

  const checkApi = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      const data = await response.json();
      setApiStatus(response.ok && data.model_loaded ? "ready" : "model");
    } catch {
      setApiStatus("offline");
    }
  }, []);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const response = await fetch(`${API_BASE}/history?limit=50`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to load history.");
      setHistory(data.items || []);
    } catch (err) {
      setHistoryError(err.message || "Unable to load history.");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    checkApi();
  }, [checkApi]);

  useEffect(() => {
    if (activeTab === "history") loadHistory();
  }, [activeTab, loadHistory]);

  function updateField(name, value) {
    setForm(current => ({ ...current, [name]: value }));
  }

  async function handlePredict(event) {
    event.preventDefault();
    setError("");
    setResult(null);
    const values = {};
    for (const field of fields) {
      const raw = form[field.name];
      if (raw === "" || !Number.isFinite(Number(raw))) {
        setError(`Please enter a valid number for ${field.label}.`);
        return;
      }
      values[field.name] = Number(raw);
    }
    if (!Number.isInteger(values.Year) || !Number.isInteger(values.nst)) {
      setError("Year and nst must be whole numbers.");
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed.");
      setResult(data);
      setApiStatus("ready");
    } catch (err) {
      setError(`${err.message || "Could not connect to the API."} Make sure the FastAPI backend is running.`);
      checkApi();
    } finally {
      setLoading(false);
    }
  }

  const predictedClass = result?.predicted_class;
  const isPositiveOne = String(predictedClass) === "1";
  const isZero = String(predictedClass) === "0";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" onClick={() => setActiveTab("predict")} aria-label="TsunamiGuard AI home">
          <span className="brand-mark">≈</span>
          <span>TsunamiGuard <b>AI</b></span>
        </a>
        <nav className="nav-tabs" aria-label="Main navigation">
          <button className={activeTab === "predict" ? "nav-tab active" : "nav-tab"} onClick={() => setActiveTab("predict")}>Predict</button>
          <button className={activeTab === "history" ? "nav-tab active" : "nav-tab"} onClick={() => setActiveTab("history")}>Prediction history</button>
        </nav>
        <div className={`api-pill ${apiStatus}`}>
          <span className="status-dot" />
          {apiStatus === "ready" ? "Model connected" : apiStatus === "checking" ? "Checking API" : apiStatus === "model" ? "Model unavailable" : "API offline"}
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-line" /> MACHINE LEARNING DEMONSTRATION</div>
            <h1>Understand the signals.<br /><span>Explore the prediction.</span></h1>
            <p>Enter earthquake-related measurements and run your trained model to view its classification and keep a record of predictions.</p>
            <div className="hero-note"><span className="shield">◇</span> Educational tool · Not an official warning system</div>
          </div>
          <div className="ocean-art" aria-hidden="true">
            <div className="orb orb-one" /><div className="orb orb-two" />
            <div className="wave wave-one" /><div className="wave wave-two" /><div className="wave wave-three" />
            <div className="ocean-label"><span className="pulse" /> MODEL-BASED ANALYSIS</div>
          </div>
        </section>

        {activeTab === "predict" ? (
          <section className="workspace">
            <div className="section-heading">
              <div><div className="eyebrow">01 / INPUT DATA</div><h2>Prediction parameters</h2><p>Enter all eight features expected by the trained model.</p></div>
              <span className="required-label">8 required inputs</span>
            </div>
            <div className="content-grid">
              <form className="panel form-panel" onSubmit={handlePredict}>
                <div className="panel-title"><span className="panel-icon">⌁</span><div><h3>Event measurements</h3><p>Use values in the same units and ranges as the training data.</p></div></div>
                <div className="field-grid">
                  {fields.map(field => (
                    <label className="field" key={field.name}>
                      <span>{field.label}<span className="asterisk"> *</span></span>
                      <input
                        type="number"
                        name={field.name}
                        value={form[field.name]}
                        onChange={event => updateField(field.name, event.target.value)}
                        step={field.step}
                        min={field.min}
                        max={field.max}
                        required
                      />
                      <small>{field.hint}</small>
                    </label>
                  ))}
                </div>
                {error && <div className="message error" role="alert"><span>!</span>{error}</div>}
                <button className="primary-button" type="submit" disabled={loading}>
                  {loading ? <><span className="spinner" /> Running model…</> : <>Predict tsunami class <span>↗</span></>}
                </button>
                <p className="form-footnote">Your values are sent to the backend model. Successful predictions are saved when Supabase is configured.</p>
              </form>

              <aside className="result-column">
                <div className="panel result-panel">
                  <div className="panel-title"><span className="panel-icon">✳</span><div><h3>Prediction result</h3><p>Output from your saved model</p></div></div>
                  {!result && !loading && (
                    <div className="empty-result">
                      <div className="empty-symbol">≈</div>
                      <h4>Ready when you are</h4>
                      <p>Complete the input form and select Predict to see the model's output here.</p>
                    </div>
                  )}
                  {loading && <div className="loading-result"><span className="large-spinner" /><h4>Analyzing inputs</h4><p>Waiting for the model response…</p></div>}
                  {result && (
                    <div className="result-content">
                      <div className="result-status">{isPositiveOne ? "CLASS 1" : isZero ? "CLASS 0" : "MODEL OUTPUT"}</div>
                      <div className="result-value">{isPositiveOne ? "Class 1" : isZero ? "Class 0" : String(predictedClass)}</div>
                      <p className="result-description">Predicted class returned by your trained model. Confirm your training target's label mapping before interpreting this as tsunami/no tsunami.</p>
                      {Array.isArray(result.probabilities) && result.probabilities.length > 0 && (
                        <div className="probability-list">
                          <h4>Model probability scores</h4>
                          {result.probabilities.map(item => (
                            <div className="prob-row" key={item.class}>
                              <div className="prob-label"><span>Class {item.class}</span><b>{(item.probability * 100).toFixed(1)}%</b></div>
                              <div className="prob-track"><span style={{ width: `${Math.max(0, Math.min(100, item.probability * 100))}%` }} /></div>
                            </div>
                          ))}
                        </div>
                      )}
                      <div className={`save-status ${result.saved_to_history ? "saved" : "not-saved"}`}>
                        {result.saved_to_history ? "✓ Saved to prediction history" : `! Not saved to history${result.history_warning ? `: ${result.history_warning}` : ""}`}
                      </div>
                      <button className="secondary-button" onClick={() => { setActiveTab("history"); }} type="button">View prediction history <span>→</span></button>
                    </div>
                  )}
                </div>
                <div className="safety-card"><span className="safety-icon">ⓘ</span><div><b>Important safety note</b><p>This project is for education and experimentation. Always follow official tsunami alerts and instructions from relevant authorities.</p></div></div>
              </aside>
            </div>
          </section>
        ) : (
          <section className="workspace history-workspace">
            <div className="section-heading">
              <div><div className="eyebrow">02 / SAVED RECORDS</div><h2>Prediction history</h2><p>Recent predictions retrieved from your Supabase database.</p></div>
              <button className="secondary-button refresh-button" onClick={loadHistory} disabled={historyLoading}>{historyLoading ? "Refreshing…" : "↻ Refresh history"}</button>
            </div>
            {historyError && <div className="message error" role="alert"><span>!</span>{historyError}<span className="history-tip">Check the backend .env values and run the SQL schema in Supabase.</span></div>}
            <div className="panel history-panel">
              {historyLoading ? <div className="history-empty"><span className="large-spinner" /><p>Loading saved predictions…</p></div> :
                history.length === 0 ? <div className="history-empty"><div className="empty-symbol">⌁</div><h3>No saved predictions yet</h3><p>Run a prediction with Supabase configured and saved records will appear here.</p></div> :
                  <div className="table-wrap"><table><thead><tr><th>Predicted class</th><th>Year</th><th>Latitude</th><th>Longitude</th><th>Depth</th><th>Gap</th><th>Sig</th><th>Created</th></tr></thead>
                    <tbody>{history.map(item => <tr key={item.id}><td><span className="class-chip">Class {item.predicted_class}</span></td><td>{item.year}</td><td>{item.latitude}</td><td>{item.longitude}</td><td>{item.depth}</td><td>{item.gap}</td><td>{item.sig}</td><td>{formatDate(item.created_at)}</td></tr>)}</tbody>
                  </table></div>}
            </div>
            <p className="history-footnote">History is loaded through FastAPI. The Supabase secret key is never sent to the browser.</p>
          </section>
        )}
      </main>
      <footer className="footer"><span>≈ TsunamiGuard AI</span><span>Educational machine-learning demonstration · Not for emergency decisions</span></footer>
    </div>
  );
}

export default App;
