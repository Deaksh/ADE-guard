import React, { useState, useEffect } from 'react';
import { fetchReportById } from "../services/api";

export default function ADEReportViewer({ vaersId }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
  if (!vaersId) {
    setReport(null);
    return;
  }
  setLoading(true);
  setError(null);
  fetchReportById(vaersId)
    .then(data => {
        console.log("Received report data:", data);
      setReport(data);
      setLoading(false);
    })
    .catch(e => {
      setError(e.message);
      setLoading(false);
    });
}, [vaersId]);

  if (loading) return <div>Loading report #{vaersId}...</div>;
  if (error) return <div style={{color: 'red'}}>Error loading report: {error}</div>;
  if (!vaersId) return <div>Select a report to view details.</div>;
  if (!report || report.text === "Not found") return <div>Report not found for VAERS ID: {vaersId}</div>;

  // Render token-wise highlights
  const renderHighlightedText = () => {
  const { text, ner_entities } = report;
  console.log("NER entities:", ner_entities);

  if (!ner_entities || ner_entities.length === 0) return <p>{text}</p>;

  const spans = ner_entities
    .filter(e =>
      typeof e.start === "number" &&
      typeof e.end === "number" &&
      e.start >= 0 &&
      e.end > e.start &&
      e.end <= text.length
    )
    .map(e => ({
      start: e.start,
      end: e.end,
      type: e.entity_group,
      text: text.substring(e.start, e.end),
    }))
    .sort((a, b) => a.start - b.start);

  if (spans.length === 0) return <p>{text}</p>;

  let lastPos = 0;
  const elements = [];
  spans.forEach(({ start, end, type, text: spanText }, i) => {
    if (start > lastPos) {
      elements.push(<span key={`txt-${i}-plain`}>{text.substring(lastPos, start)}</span>);
    }
    elements.push(
      <mark
        key={`mark-${i}`}
        style={{
          backgroundColor: type === "ADE" ? "#ffcccc" : "#cce5ff",
          cursor: "pointer",
        }}
        title={`${type} entity`}
      >
        {spanText}
      </mark>
    );
    lastPos = end;
  });
  if (lastPos < text.length) {
    elements.push(<span key="txt-end">{text.substring(lastPos)}</span>);
  }
  return <p style={{ whiteSpace: "pre-wrap" }}>{elements}</p>;
};


  return (
    <div>
      <h2>VAERS Report #{report.VAERS_ID}</h2>
      <div><strong>Severity:</strong> {report.predicted_severity?.[0] || report.predicted_severity || "Unknown"} {/* Adjust if backend returns differently */}</div>
      <h3>Symptom Text with NER Highlights:</h3>
      <div style={{ border: "1px solid #ccc", borderRadius: 4, padding: 10, marginBottom: 20 }}>
        {renderHighlightedText()}
      </div>
      {report.severity_explanation && (
        <div>
          <h3>Severity Explanation (SHAP values):</h3>
          <pre style={{ maxHeight: "200px", overflowY: "scroll", background: "#f9f9f9", padding: 10 }}>
            Tokens: {report.severity_explanation.tokens.join(" | ")}
            <br />
            Values: {report.severity_explanation.shap_values.map(v => v.toFixed(3)).join(" | ")}
          </pre>
        </div>
      )}
    </div>
  );
}
