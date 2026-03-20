"use client";

import { useMemo, useState } from "react";
import axios from "axios";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type Entity = {
  text: string;
  label: string;
  start: number;
  end: number;
  score: number;
};

type Severity = {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
};

type ExplainFeature = { token: string; weight: number };

export default function AdeDashboard() {
  const [text, setText] = useState(
    "Patient developed severe chest pain and dizziness after the Pfizer COVID-19 vaccine."
  );
  const [entities, setEntities] = useState<Entity[]>([]);
  const [severity, setSeverity] = useState<Severity | null>(null);
  const [analysis, setAnalysis] = useState<any[]>([]);
  const [explain, setExplain] = useState<any>(null);
  const [clusters, setClusters] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const highlighted = useMemo(() => buildSegments(text, entities), [text, entities]);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const [nerRes, sevRes, analyzeRes, explainRes] = await Promise.all([
        axios.post(`${API_BASE}/api/v1/ner`, { text }),
        axios.post(`${API_BASE}/api/v1/severity`, { text }),
        axios.post(`${API_BASE}/api/v1/analyze`, { text }),
        axios.post(`${API_BASE}/api/v1/explain/severity`, { text }),
      ]);
      setEntities(nerRes.data.entities || []);
      setSeverity(sevRes.data);
      setAnalysis(analyzeRes.data.results || []);
      setExplain(explainRes.data || null);
    } finally {
      setLoading(false);
    }
  };

  const handleClusters = async () => {
    const res = await axios.get(
      `${API_BASE}/api/v1/clusters?max_records=500&min_cluster_size=15&include_points=1`
    );
    setClusters(res.data);
  };

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="mx-auto max-w-6xl space-y-10">
        <header className="space-y-4">
          <div className="inline-flex items-center gap-2 badge bg-sky text-ink">
            ADEGuard • AI Safety Console
          </div>
          <h1 className="text-4xl md:text-5xl font-display font-semibold text-ink">
            Real-time ADE Intelligence
          </h1>
          <p className="text-slate max-w-2xl">
            Detect adverse drug events, cluster symptom variants across ages, and
            interpret severity signals with explainable AI.
          </p>
        </header>

        <section className="grid md:grid-cols-[1.2fr_0.8fr] gap-6">
          <div className="card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="font-display text-xl">Narrative Intake</h2>
              <button
                className="px-4 py-2 rounded-full bg-ink text-sand text-sm font-semibold"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Run Analysis"}
              </button>
            </div>
            <textarea
              className="w-full min-h-[140px] rounded-2xl border border-slate/20 bg-white/70 p-4 text-sm focus:outline-none focus:ring-2 focus:ring-sea"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-widest text-slate/70">Token Highlights</p>
              <div className="rounded-2xl border border-slate/10 bg-white p-4 text-sm leading-6">
                {highlighted.map((segment, idx) => (
                  <span
                    key={`${segment.text}-${idx}`}
                    className={segment.className}
                  >
                    {segment.text}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="card p-6 space-y-6">
            <div>
              <h2 className="font-display text-xl">Severity Signal</h2>
              <p className="text-sm text-slate">Classifier output with confidence.</p>
            </div>
            <div className="rounded-2xl border border-slate/10 bg-white p-4">
              {severity ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">{severity.label}</span>
                    <span className="text-sm text-slate">
                      {(severity.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="space-y-2">
                    {Object.entries(severity.probabilities || {}).map(([label, value]) => (
                      <div key={label} className="space-y-1">
                        <div className="flex justify-between text-xs text-slate">
                          <span>{label}</span>
                          <span>{(value * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 rounded-full bg-sky">
                          <div
                            className="h-2 rounded-full bg-sea"
                            style={{ width: `${value * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-sm text-slate">Run analysis to view severity.</p>
              )}
            </div>
          </div>
        </section>

        <section className="grid md:grid-cols-2 gap-6">
          <div className="card p-6 space-y-4">
            <h2 className="font-display text-xl">ADE Spans & Modifiers</h2>
            <div className="space-y-3 text-sm">
              {analysis.length === 0 && (
                <p className="text-slate">No ADE spans extracted yet.</p>
              )}
              {analysis.map((item, idx) => (
                <div key={`${item.entity}-${idx}`} className="flex items-center justify-between">
                  <div>
                    <p className="font-semibold">{item.entity}</p>
                    <p className="text-xs text-slate">{item.label}</p>
                  </div>
                  <span className="badge bg-coral/20 text-ink">
                    {Array.isArray(item.severity)
                      ? `${item.severity[0]} ${(item.severity[1] * 100).toFixed(1)}%`
                      : item.severity}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-6 space-y-4">
            <h2 className="font-display text-xl">Explainability</h2>
            <p className="text-sm text-slate">
              LIME and SHAP highlight tokens driving the severity decision.
            </p>
            <div className="grid gap-4">
              <ExplainList title="LIME" features={explain?.lime?.features} error={explain?.lime?.error} />
              <ExplainList title="SHAP" features={explain?.shap?.features} error={explain?.shap?.error} />
            </div>
          </div>
        </section>

        <section className="card p-6 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-display text-xl">Age-aware Cluster Explorer</h2>
              <p className="text-sm text-slate">
                HDBSCAN clusters of ADE mentions with modifiers and age bands.
              </p>
            </div>
            <button
              className="px-4 py-2 rounded-full bg-sea text-white text-sm font-semibold"
              onClick={handleClusters}
            >
              Load Clusters
            </button>
          </div>
          {clusters ? (
            <div className="grid md:grid-cols-[1.3fr_0.7fr] gap-6">
              <div className="h-72 rounded-2xl border border-slate/10 bg-white p-4">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <XAxis type="number" dataKey="x" hide />
                    <YAxis type="number" dataKey="y" hide />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                    <Scatter data={clusters.points || []} fill="#1d6f8a" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-3 text-sm">
                {(clusters.clusters || []).slice(0, 8).map((c: any, idx: number) => (
                  <div key={`${c.cluster_id}-${idx}`} className="rounded-2xl border border-slate/10 bg-white p-4">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">Cluster {c.cluster_id}</span>
                      <span className="badge bg-sky text-ink">{c.count} mentions</span>
                    </div>
                    <p className="text-xs text-slate">{c.age_group} • {c.modifier}</p>
                    <p className="text-xs text-slate mt-2">{(c.symptoms || []).join(", ")}</p>
                    {c.modifier_counts && (
                      <p className="text-[11px] text-slate mt-2">
                        Modifiers: {Object.entries(c.modifier_counts).map(([k, v]) => `${k} (${v})`).join(", ")}
                      </p>
                    )}
                    {c.top_examples && c.top_examples.length > 0 && (
                      <div className="mt-2 space-y-1 text-[11px] text-slate">
                        {c.top_examples.map((ex: string, exIdx: number) => (
                          <p key={`${c.cluster_id}-ex-${exIdx}`}>“{ex}”</p>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate">Load clusters to visualize age-aware patterns.</p>
          )}
        </section>
      </div>
    </div>
  );
}

function ExplainList({ title, features, error }: { title: string; features?: ExplainFeature[]; error?: string }) {
  return (
    <div className="rounded-2xl border border-slate/10 bg-white p-4">
      <div className="flex items-center justify-between">
        <span className="font-semibold">{title}</span>
        <span className="text-xs text-slate">Top tokens</span>
      </div>
      <div className="mt-3 space-y-2">
        {error ? (
          <p className="text-xs text-coral">{error}</p>
        ) : features && features.length > 0 ? (
          features.map((f, idx) => (
            <div key={`${f.token}-${idx}`} className="flex items-center justify-between text-xs">
              <span className="truncate">{f.token}</span>
              <span className="text-slate">{f.weight.toFixed(3)}</span>
            </div>
          ))
        ) : (
          <p className="text-xs text-slate">Run analysis to populate.</p>
        )}
      </div>
    </div>
  );
}

function buildSegments(text: string, entities: Entity[]) {
  if (!text) return [] as { text: string; className?: string }[];
  const sorted = [...entities].sort((a, b) => a.start - b.start);
  const segments: { text: string; className?: string }[] = [];
  let cursor = 0;

  for (const ent of sorted) {
    if (ent.start > cursor) {
      segments.push({ text: text.slice(cursor, ent.start) });
    }
    const slice = text.slice(ent.start, ent.end);
    segments.push({ text: slice, className: labelClass(ent.label) });
    cursor = ent.end;
  }

  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor) });
  }

  return segments;
}

function labelClass(label: string) {
  const clean = label.toUpperCase().replace("B-", "").replace("I-", "");
  if (clean === "ADE" || clean === "ADR" || clean === "ADVERSE_EVENT") return "highlight-ade";
  if (clean === "DRUG") return "highlight-drug";
  if (clean === "AGE") return "highlight-age";
  return "";
}
