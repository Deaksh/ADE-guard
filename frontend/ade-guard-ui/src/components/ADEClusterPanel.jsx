import React, { useEffect, useState } from "react";
import { fetchClusters } from "../services/api";

export default function ADEClusterPanel({ onClusterSelect }) {
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchClusters()
      .then(data => {
        setClusters(data.clusters || []);
        setLoading(false);
      })
      .catch(e => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading clusters...</div>;
  if (error) return <div style={{ color: "red" }}>Error: {error}</div>;
  if (!clusters.length) return <div>No clusters found.</div>;

  return (
    <div style={{ marginTop: 30 }}>
      <h2>Symptom Clusters by Modifier & Age</h2>
      <table style={{ borderCollapse: "collapse", width: "100%", marginTop: 12 }}>
        <thead>
          <tr style={{ background: "#eee" }}>
            <th>Cluster ID</th>
            <th>Age Group</th>
            <th>Modifier</th>
            <th>Symptoms</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {clusters.map((c) => (
            <tr key={c.cluster_id}>
              <td>{c.cluster_id}</td>
              <td>{c.age_group}</td>
              <td style={{ color:
                c.modifier === "Severe"
                ? "#d32f2f"
                : c.modifier === "Moderate"
                ? "#ed6c02"
                : c.modifier === "Mild"
                ? "#2e7d32"
                : "#757575"
              }}>{c.modifier}</td>
              <td>{c.symptoms.join(", ")}</td>
              <td>
                <button
                  onClick={() => onClusterSelect && onClusterSelect(c)}
                  style={{ padding: "2px 8px" }}
                >
                  View Reports
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
