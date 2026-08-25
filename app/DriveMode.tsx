"use client";

import { BellOff, Minimize, ShieldCheck, Square, Timer, Zap } from "lucide-react";
import { formatDuration } from "@/lib/detection/core.mjs";

type RiskState = "focused" | "caution" | "warning" | "danger";

type DriveModeProps = {
  risk: number;
  state: RiskState;
  label: string;
  action: string;
  recommendation: string;
  sessionSeconds: number;
  confidence: number;
  faceFound: boolean;
  calibrating: boolean;
  calibrationProgress: number;
  counterfactual: string | null;
  breakDue: boolean;
  snoozeRemaining: number;
  wakeLockHeld: boolean;
  onSnooze: () => void;
  onStop: () => void;
  onExit: () => void;
};

/**
 * The at-a-glance view for a phone on a windshield mount.
 *
 * Everything here is sized to be read in well under a second and operated with
 * one thumb. The camera preview is deliberately absent: a driver watching a
 * video of themselves is the exact failure this project exists to prevent.
 */
export default function DriveMode({
  risk,
  state,
  label,
  action,
  recommendation,
  sessionSeconds,
  confidence,
  faceFound,
  calibrating,
  calibrationProgress,
  counterfactual,
  breakDue,
  snoozeRemaining,
  wakeLockHeld,
  onSnooze,
  onStop,
  onExit,
}: DriveModeProps) {
  const unsure = confidence < 0.45;

  return (
    <section className={`drive-mode state-${state}`} aria-label="Drive mode">
      <header className="drive-top">
        <span className="drive-clock">
          <Timer size={14} /> {formatDuration(sessionSeconds)}
        </span>
        <span className="drive-badges">
          {wakeLockHeld && (
            <span className="drive-badge" title="Screen kept awake">
              <Zap size={13} />
            </span>
          )}
          <span className="drive-badge" title="Processing on this device">
            <ShieldCheck size={13} />
          </span>
        </span>
        <button className="drive-exit" onClick={onExit} aria-label="Leave drive mode">
          <Minimize size={18} />
        </button>
      </header>

      <div className="drive-core" role="status" aria-live="polite">
        {calibrating ? (
          <>
            <p className="drive-caption">Calibrating</p>
            <strong className="drive-risk">{Math.round(calibrationProgress * 100)}%</strong>
            <p className="drive-status">Look straight ahead</p>
          </>
        ) : (
          <>
            <p className="drive-caption">Attention risk</p>
            <strong className="drive-risk">{risk}</strong>
            <p className="drive-status">{label}</p>
            <p className="drive-action">{unsure ? "Camera view is unclear" : action}</p>
          </>
        )}
      </div>

      <div className="drive-notes">
        {!calibrating && !faceFound && <p className="drive-note">Face not visible</p>}
        {!calibrating && unsure && faceFound && (
          <p className="drive-note">Low confidence. {recommendation}</p>
        )}
        {breakDue && <p className="drive-note">Two hours driving. Plan a break.</p>}
        {counterfactual && !unsure && <p className="drive-note subtle">{counterfactual}</p>}
      </div>

      <footer className="drive-actions">
        <button
          className={`drive-snooze ${snoozeRemaining > 0 ? "is-active" : ""}`}
          onClick={onSnooze}
          aria-label={snoozeRemaining > 0 ? "Alerts snoozed" : "Snooze alerts for five minutes"}
        >
          <BellOff size={18} />
          <span>
            {snoozeRemaining > 0 ? formatDuration(Math.ceil(snoozeRemaining / 1000)) : "Snooze"}
          </span>
        </button>
        <button className="drive-stop" onClick={onStop}>
          <Square size={20} fill="currentColor" />
          <span>End session</span>
        </button>
      </footer>
    </section>
  );
}
