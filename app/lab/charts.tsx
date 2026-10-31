"use client";

/**
 * Small, dependency-free SVG charts for the Model Lab. Every number drawn here
 * is read from the JSON the training scripts write to public/results, so the page
 * can never disagree with the experiments.
 */

import { useId } from "react";

const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

export type BarRow = {
  key: string;
  label: string;
  value: number;
  low?: number;
  high?: number;
  highlight?: boolean;
};

export type ReferenceLine = { label: string; value: number; tone?: "warn" | "muted" };

/** Horizontal bars with optional 95% interval whiskers and labelled reference lines. */
export function BarChart({
  rows,
  references = [],
  format = pct,
  axisLabel,
  max = 1,
}: {
  rows: BarRow[];
  references?: ReferenceLine[];
  format?: (value: number) => string;
  axisLabel: string;
  max?: number;
}) {
  const left = 168;
  const right = 84;
  const width = 720;
  const rowHeight = 34;
  const top = references.length ? 34 : 12;
  const height = top + rows.length * rowHeight + 34;
  const x = (value: number) => left + (Math.max(0, value) / max) * (width - left - right);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * max);
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={axisLabel}>
      {ticks.map((tick) => (
        <g key={tick}>
          <line x1={x(tick)} x2={x(tick)} y1={top - 6} y2={height - 28} className="chart-grid" />
          <text x={x(tick)} y={height - 12} className="chart-tick" textAnchor="middle">
            {format(tick)}
          </text>
        </g>
      ))}
      {references.map((reference, index) => (
        <g key={reference.label}>
          <line
            x1={x(reference.value)}
            x2={x(reference.value)}
            y1={top - 6}
            y2={height - 28}
            className={`chart-ref ${reference.tone === "warn" ? "is-warn" : ""}`}
          />
          <text
            x={x(reference.value)}
            y={12 + (index % 2) * 13}
            textAnchor="middle"
            className={`chart-ref-label ${reference.tone === "warn" ? "is-warn" : ""}`}
          >
            {reference.label} {format(reference.value)}
          </text>
        </g>
      ))}
      {rows.map((row, index) => {
        const y = top + index * rowHeight;
        return (
          <g key={row.key}>
            <text x={left - 12} y={y + 18} textAnchor="end" className="chart-label">
              {row.label}
            </text>
            <rect
              x={left}
              y={y + 6}
              width={Math.max(1, x(row.value) - left)}
              height={18}
              rx={4}
              className={row.highlight ? "chart-bar is-accent" : "chart-bar"}
            >
              <title>{`${row.label}: ${format(row.value)}`}</title>
            </rect>
            {row.low !== undefined && row.high !== undefined && (
              <g className="chart-whisker">
                <line x1={x(row.low)} x2={x(row.high)} y1={y + 15} y2={y + 15} />
                <line x1={x(row.low)} x2={x(row.low)} y1={y + 10} y2={y + 20} />
                <line x1={x(row.high)} x2={x(row.high)} y1={y + 10} y2={y + 20} />
              </g>
            )}
            {/* Values sit in their own column so reference lines never cross them. */}
            <text x={width - 4} y={y + 19} textAnchor="end" className="chart-value">
              {format(row.value)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export type Curve = { key: string; label: string; fpr: number[]; tpr: number[]; auc: number; highlight?: boolean };

/** ROC curves; the highlighted curve takes the accent, the rest stay quiet. */
export function RocChart({ curves, title }: { curves: Curve[]; title: string }) {
  const size = 300;
  const pad = 40;
  const p = (value: number) => pad + value * (size - pad - 12);
  const q = (value: number) => size - pad - value * (size - pad - 12);
  const sorted = [...curves].sort((a, b) => Number(a.highlight) - Number(b.highlight));
  return (
    <div className="roc">
      <svg className="chart" viewBox={`0 0 ${size} ${size}`} role="img" aria-label={title}>
        {[0, 0.5, 1].map((tick) => (
          <g key={tick}>
            <line x1={p(tick)} x2={p(tick)} y1={q(0)} y2={q(1)} className="chart-grid" />
            <line x1={p(0)} x2={p(1)} y1={q(tick)} y2={q(tick)} className="chart-grid" />
            <text x={p(tick)} y={size - pad + 16} textAnchor="middle" className="chart-tick">
              {tick}
            </text>
            <text x={pad - 8} y={q(tick) + 4} textAnchor="end" className="chart-tick">
              {tick}
            </text>
          </g>
        ))}
        <line x1={p(0)} y1={q(0)} x2={p(1)} y2={q(1)} className="chart-ref" />
        {sorted.map((curve, index) => (
          <polyline
            key={curve.key}
            fill="none"
            className={curve.highlight ? "roc-line is-accent" : `roc-line tone-${index % 4}`}
            points={curve.fpr.map((f, i) => `${p(f)},${q(curve.tpr[i])}`).join(" ")}
          >
            <title>{`${curve.label}: AUC ${curve.auc.toFixed(3)}`}</title>
          </polyline>
        ))}
        <text x={(p(0) + p(1)) / 2} y={size - 6} textAnchor="middle" className="chart-axis">
          False positive rate
        </text>
        <text
          transform={`translate(12 ${(q(0) + q(1)) / 2}) rotate(-90)`}
          textAnchor="middle"
          className="chart-axis"
        >
          True positive rate
        </text>
      </svg>
      <ol className="legend">
        {[...curves]
          .sort((a, b) => b.auc - a.auc)
          .map((curve) => (
            <li key={curve.key} className={curve.highlight ? "is-accent" : ""}>
              <span className="legend-swatch" data-highlight={curve.highlight ? "yes" : "no"} />
              <span>{curve.label}</span>
              <strong>{curve.auc.toFixed(3)}</strong>
            </li>
          ))}
      </ol>
    </div>
  );
}

/** Confusion matrix shaded by row share, counts and shares printed in each cell. */
export function Confusion({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const cell = 74;
  const left = 96;
  const top = 34;
  const size = left + cell * labels.length + 8;
  return (
    <svg className="chart confusion" viewBox={`0 0 ${size} ${top + cell * labels.length + 36}`} role="img" aria-label="Confusion matrix">
      {labels.map((label, j) => (
        <text key={label} x={left + j * cell + cell / 2} y={top - 12} textAnchor="middle" className="chart-label">
          {label}
        </text>
      ))}
      {matrix.map((row, i) => {
        const total = row.reduce((sum, value) => sum + value, 0) || 1;
        return (
          <g key={labels[i]}>
            <text x={left - 10} y={top + i * cell + cell / 2 + 4} textAnchor="end" className="chart-label">
              {labels[i]}
            </text>
            {row.map((value, j) => {
              const share = value / total;
              return (
                <g key={j}>
                  <rect
                    x={left + j * cell + 2}
                    y={top + i * cell + 2}
                    width={cell - 4}
                    height={cell - 4}
                    rx={8}
                    className="confusion-cell"
                    style={{ fillOpacity: 0.08 + share * 0.82 }}
                    data-diagonal={i === j ? "yes" : "no"}
                  />
                  <text
                    x={left + j * cell + cell / 2}
                    y={top + i * cell + cell / 2 - 2}
                    textAnchor="middle"
                    className={share > 0.55 ? "confusion-text is-inverse" : "confusion-text"}
                  >
                    {pct(share, 0)}
                  </text>
                  <text
                    x={left + j * cell + cell / 2}
                    y={top + i * cell + cell / 2 + 14}
                    textAnchor="middle"
                    className={share > 0.55 ? "confusion-count is-inverse" : "confusion-count"}
                  >
                    n={value}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
      <text x={left + (cell * labels.length) / 2} y={top + cell * labels.length + 26} textAnchor="middle" className="chart-axis">
        Predicted (rows are the truth)
      </text>
    </svg>
  );
}

/** Reliability diagram: observed accuracy against stated confidence. */
export function Reliability({
  series,
}: {
  series: { key: string; label: string; points: { confidence: number; accuracy: number }[]; highlight?: boolean }[];
}) {
  const size = 260;
  const pad = 36;
  const p = (value: number) => pad + value * (size - pad - 10);
  const q = (value: number) => size - pad - value * (size - pad - 10);
  return (
    <svg className="chart" viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Reliability diagram">
      {[0, 0.5, 1].map((tick) => (
        <g key={tick}>
          <line x1={p(tick)} x2={p(tick)} y1={q(0)} y2={q(1)} className="chart-grid" />
          <line x1={p(0)} x2={p(1)} y1={q(tick)} y2={q(tick)} className="chart-grid" />
          <text x={p(tick)} y={size - pad + 15} textAnchor="middle" className="chart-tick">{tick}</text>
          <text x={pad - 6} y={q(tick) + 4} textAnchor="end" className="chart-tick">{tick}</text>
        </g>
      ))}
      <line x1={p(0)} y1={q(0)} x2={p(1)} y2={q(1)} className="chart-ref" />
      {series.map((s) => (
        <g key={s.key} className={s.highlight ? "rel is-accent" : "rel"}>
          <polyline fill="none" points={s.points.map((pt) => `${p(pt.confidence)},${q(pt.accuracy)}`).join(" ")} />
          {s.points.map((pt) => (
            <circle key={pt.confidence} cx={p(pt.confidence)} cy={q(pt.accuracy)} r={3.2}>
              <title>{`${s.label}: confidence ${pct(pt.confidence)}, accuracy ${pct(pt.accuracy)}`}</title>
            </circle>
          ))}
        </g>
      ))}
      <text x={(p(0) + p(1)) / 2} y={size - 4} textAnchor="middle" className="chart-axis">Stated confidence</text>
      <text transform={`translate(11 ${(q(0) + q(1)) / 2}) rotate(-90)`} textAnchor="middle" className="chart-axis">
        Observed accuracy
      </text>
    </svg>
  );
}

/** Live sparkline on a fixed 0..100 scale. */
export function Sparkline({ values, label, tone = "accent" }: { values: number[]; label: string; tone?: "accent" | "muted" | "warn" }) {
  const id = useId();
  const width = 240;
  const height = 56;
  const points = values.map((value, index) => {
    const x = values.length > 1 ? (index / (values.length - 1)) * width : width;
    return `${x},${height - (Math.min(100, Math.max(0, value)) / 100) * (height - 4) - 2}`;
  });
  return (
    <svg className={`spark tone-${tone}`} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} preserveAspectRatio="none">
      <defs>
        <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="currentColor" stopOpacity="0.22" />
          <stop offset="1" stopColor="currentColor" stopOpacity="0" />
        </linearGradient>
      </defs>
      <line x1="0" x2={width} y1={height - (54 / 100) * (height - 4) - 2} y2={height - (54 / 100) * (height - 4) - 2} className="spark-ref" />
      {points.length > 1 && (
        <>
          <polygon points={`0,${height} ${points.join(" ")} ${width},${height}`} fill={`url(#${id})`} />
          <polyline points={points.join(" ")} fill="none" stroke="currentColor" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        </>
      )}
    </svg>
  );
}
