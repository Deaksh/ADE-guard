"use client";

import { useEffect, useMemo, useRef, useState } from "react";

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
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 600, height: 360 });
  const [hovered, setHovered] = useState<Point | null>(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!wrapRef.current) return;
    const el = wrapRef.current;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      setSize({ width: rect.width, height: rect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const bounds = useMemo(() => {
    if (!points.length) {
      return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    }
    let minX = points[0].x, maxX = points[0].x, minY = points[0].y, maxY = points[0].y;
    for (const p of points) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    const padX = (maxX - minX) * 0.08 || 1;
    const padY = (maxY - minY) * 0.08 || 1;
    return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
  }, [points]);

  const scale = (x: number, min: number, max: number, sizePx: number) => {
    if (max === min) return sizePx / 2;
    return ((x - min) / (max - min)) * sizePx;
  };

  return (
    <div ref={wrapRef} className="relative h-full w-full">
      <svg width={size.width} height={size.height} className="rounded-2xl bg-white/60">
        {points.map((p, idx) => {
          const cx = scale(p.x, bounds.minX, bounds.maxX, size.width);
          const cy = size.height - scale(p.y, bounds.minY, bounds.maxY, size.height);
          const fill = MODIFIER_COLORS[p.modifier] || MODIFIER_COLORS.Unknown;
          return (
            <circle
              key={`${p.ade}-${idx}`}
              cx={cx}
              cy={cy}
              r={5}
              fill={fill}
              opacity={0.85}
              onMouseEnter={(e) => {
                setHovered(p);
                const rect = (e.target as SVGCircleElement).getBoundingClientRect();
                setHoverPos({ x: rect.left + rect.width / 2, y: rect.top });
              }}
              onMouseLeave={() => setHovered(null)}
            />
          );
        })}
      </svg>

      {hovered && (
        <div
          className="pointer-events-none fixed z-50 rounded-xl bg-white border border-slate/20 p-3 text-xs shadow"
          style={{ left: hoverPos.x + 12, top: hoverPos.y - 12 }}
        >
          <div className="font-semibold">{hovered.ade}</div>
          <div className="text-slate">{hovered.modifier} • {hovered.age_group}</div>
        </div>
      )}
    </div>
  );
}
