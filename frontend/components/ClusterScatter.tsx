"use client";

import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip } from "recharts";

const MODIFIER_COLORS: Record<string, string> = {
  Severe: "#f26b5b",
  Moderate: "#f7b84f",
  Mild: "#59d3b8",
  Unknown: "#1d6f8a",
};

type Point = {
  x: number;
  y: number;
  ade: string;
  modifier: string;
  age_group: string;
};

export default function ClusterScatter({ points }: { points: Point[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart>
        <XAxis type="number" dataKey="x" hide />
        <YAxis type="number" dataKey="y" hide />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          content={({ payload }) => {
            if (!payload || payload.length === 0) return null;
            const p = payload[0].payload as Point;
            return (
              <div className="rounded-xl bg-white border border-slate/20 p-3 text-xs shadow">
                <div className="font-semibold">{p.ade}</div>
                <div className="text-slate">{p.modifier} • {p.age_group}</div>
              </div>
            );
          }}
        />
        {Object.entries(MODIFIER_COLORS).map(([modifier, color]) => (
          <Scatter
            key={modifier}
            data={points.filter((p) => p.modifier === modifier)}
            fill={color}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
