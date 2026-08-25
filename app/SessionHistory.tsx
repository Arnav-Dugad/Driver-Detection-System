"use client";

import { Trash2, X } from "lucide-react";
import { formatDuration } from "@/lib/detection/core.mjs";
import { type StoredSession, summarizeSessions } from "./sessionStore";

const hourLabel = (hour: number | null) => {
  if (hour === null) return "--";
  const suffix = hour < 12 ? "am" : "pm";
  const twelve = hour % 12 === 0 ? 12 : hour % 12;
  return `${twelve}${suffix}`;
};

const riskTone = (risk: number) =>
  risk >= 76 ? "danger" : risk >= 54 ? "warning" : risk >= 28 ? "caution" : "focused";

/**
 * The local session journal, finally visible.
 *
 * Every number here was already being written to this browser; showing the
 * trend is what makes it useful to the person it describes.
 */
export default function SessionHistory({
  sessions,
  onClear,
  onClose,
}: {
  sessions: StoredSession[];
  onClear: () => void;
  onClose: () => void;
}) {
  const trends = summarizeSessions(sessions);

  return (
    <div className="modal-backdrop">
      <section className="history-modal" role="dialog" aria-modal="true" aria-label="Session history">
        <div className="drawer-header">
          <h2>Your sessions</h2>
          <button className="icon-button" onClick={onClose} aria-label="Close session history">
            <X size={18} />
          </button>
        </div>

        {sessions.length ? (
          <>
            <div className="history-trends">
              <div><strong>{trends.count}</strong><span>Sessions</span></div>
              <div><strong>{trends.totalMinutes}</strong><span>Minutes</span></div>
              <div><strong>{trends.averagePeakRisk}</strong><span>Avg peak risk</span></div>
              <div><strong>{trends.alertsPerHour}</strong><span>Alerts / hour</span></div>
            </div>

            {trends.worstHour !== null && trends.count > 2 && (
              <p className="history-insight">
                Your risk runs highest around <strong>{hourLabel(trends.worstHour)}</strong> and
                lowest around <strong>{hourLabel(trends.safestHour)}</strong>.
              </p>
            )}

            <div className="history-list">
              {sessions.slice(0, 12).map((session) => {
                const when = new Date(session.endedAt);
                return (
                  <div className="history-row" key={session.id ?? session.endedAt}>
                    <div>
                      <strong>
                        {when.toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                        {session.mode === "demo" ? " / demo" : ""}
                      </strong>
                      <small>
                        {when.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}
                        {" - "}
                        {formatDuration(session.durationSeconds)}
                      </small>
                    </div>
                    <div className="history-counts">
                      <span>{session.stats?.yawns ?? 0} yawns</span>
                      <span>{session.stats?.alerts ?? 0} alerts</span>
                    </div>
                    <span className={`history-peak tone-${riskTone(session.stats?.maxRisk ?? 0)}`}>
                      {session.stats?.maxRisk ?? 0}
                    </span>
                  </div>
                );
              })}
            </div>

            <button className="drawer-action danger" onClick={onClear}>
              <Trash2 size={15} /> Clear history
            </button>
          </>
        ) : (
          <div className="history-empty">
            <strong>No sessions yet</strong>
            <p>Finish a monitoring session and its numbers will appear here.</p>
          </div>
        )}

        <p className="history-privacy">
          Stored in this browser only. Counts and durations, never video.
        </p>
      </section>
    </div>
  );
}
