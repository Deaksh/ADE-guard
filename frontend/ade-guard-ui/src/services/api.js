const BASE_URL = "https://6963b9a0d911.ngrok-free.app";
// const BASE_URL = "http://localhost:8000/api/v1";

export async function fetchReportById(vaersId) {
  if (!vaersId || isNaN(vaersId)) {
    throw new Error("Invalid VAERS ID");
  }
  const res = await fetch(`${BASE_URL}/api/v1/report/${vaersId}?explain=true`);
  if (!res.ok) throw new Error(`Failed to fetch report for VAERS ID ${vaersId}`);
  return res.json();
}

export async function fetchClusters(limit = 20) {
  const res = await fetch(`${BASE_URL}/routes/cluster?limit=${limit}`);
  if (!res.ok) throw new Error("Failed to fetch clusters");
  return res.json();
}
