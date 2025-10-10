// /Users/deakshshetty/Documents/ADE-Guard/frontend/ade-guard-ui/src/components/Dashboard.jsx
import React, { useState } from "react";
import ADEReportViewer from "./ADEReportViewer";
import ADEClusterPanel from "./ADEClusterPanel";

export default function Dashboard() {
  const [inputVaersId, setInputVaersId] = useState("");
  const [vaersId, setVaersId] = useState(null);

  const handleInputChange = (e) => setInputVaersId(e.target.value);

  const handleLoadClick = () => {
    const val = inputVaersId.trim();
    if (!val || !/^\d+$/.test(val)){
      alert("Please enter a valid numeric VAERS ID");
      return;
    }
    setVaersId(parseInt(val, 10));
  };

  // Later: filter reports by cluster, for now just alert
  const handleClusterSelect = (cluster) => {
    alert(
      `Selected cluster: ${cluster.cluster_id}\nModifier: ${cluster.modifier}\nAge group: ${cluster.age_group}`
    );
  };

  return (
    <div style={{ maxWidth: 900, margin: "auto", padding: 20 }}>
      <h1>ADEGuard Dashboard</h1>
      <div style={{ marginBottom: 20 }}>
        <label>
          Enter VAERS ID:
          <input
            type="text"
            value={inputVaersId}
            onChange={handleInputChange}
            placeholder="e.g. 2827963"
            style={{ marginLeft: 8, padding: 4, width: 150 }}
          />
        </label>
        <button
          onClick={handleLoadClick}
          style={{ marginLeft: 8, padding: "6px 12px" }}
        >
          Load Report
        </button>
      </div>
      {/* Only ONE viewer needed */}
      <ADEReportViewer vaersId={vaersId} />
      <ADEClusterPanel onClusterSelect={handleClusterSelect} />
    </div>
  );
}
