"use client";

/**
 * Local, numeric-only session journal.
 *
 * Nothing here leaves the browser and nothing here is an image: a stored
 * session is counts, durations, and risk states. It exists so the driver can
 * see their own trend over time, which is the part of the data that is
 * actually useful to them.
 */

export type StoredStats = {
  blinks: number;
  yawns: number;
  distractions: number;
  phoneEvents: number;
  alerts: number;
  maxRisk: number;
};

export type StoredSession = {
  id: string;
  endedAt: string;
  durationSeconds: number;
  mode: "camera" | "demo";
  stats: StoredStats;
  averageRisk: number;
  startHour: number;
};

const SESSION_KEY = "driver-detection-session-history";
const SUBJECT_KEY = "driver-detection-subject-id";
const MAX_SESSIONS = 40;

export function readSessions(): StoredSession[] {
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Sessions written by earlier versions may be missing newer fields.
    return parsed.filter(
      (entry): entry is StoredSession =>
        Boolean(entry) && typeof (entry as StoredSession).endedAt === "string",
    );
  } catch {
    return [];
  }
}

export function saveSession(session: StoredSession): StoredSession[] {
  const next = [session, ...readSessions()].slice(0, MAX_SESSIONS);
  try {
    window.localStorage.setItem(SESSION_KEY, JSON.stringify(next));
  } catch {
    // Reporting is a convenience; monitoring must never depend on storage.
  }
  return next;
}

export function clearSessions() {
  try {
    window.localStorage.removeItem(SESSION_KEY);
  } catch {
    // Nothing to do if storage is unavailable.
  }
}

/**
 * A random, local-only participant tag for exported research CSVs.
 *
 * `train_fusion.py` needs a grouping key to keep each person inside one split.
 * This is a random token, never a name or an identifier tied to the person.
 */
export function subjectId(): string {
  try {
    const existing = window.localStorage.getItem(SUBJECT_KEY);
    if (existing) return existing;
    const created = `subject-${Math.random().toString(36).slice(2, 8)}`;
    window.localStorage.setItem(SUBJECT_KEY, created);
    return created;
  } catch {
    return "subject-local";
  }
}

export type SessionTrends = {
  count: number;
  totalMinutes: number;
  averagePeakRisk: number;
  alertsPerHour: number;
  worstHour: number | null;
  safestHour: number | null;
};

/** Aggregate the journal into the handful of numbers worth showing. */
export function summarizeSessions(sessions: StoredSession[]): SessionTrends {
  if (!sessions.length) {
    return {
      count: 0,
      totalMinutes: 0,
      averagePeakRisk: 0,
      alertsPerHour: 0,
      worstHour: null,
      safestHour: null,
    };
  }

  const totalSeconds = sessions.reduce((sum, item) => sum + (item.durationSeconds || 0), 0);
  const totalAlerts = sessions.reduce((sum, item) => sum + (item.stats?.alerts ?? 0), 0);
  const peaks = sessions.map((item) => item.stats?.maxRisk ?? 0);

  // Group peak risk by the hour a session started to expose time-of-day pattern.
  const byHour = new Map<number, number[]>();
  for (const session of sessions) {
    const hour = Number.isFinite(session.startHour)
      ? session.startHour
      : new Date(session.endedAt).getHours();
    byHour.set(hour, [...(byHour.get(hour) ?? []), session.stats?.maxRisk ?? 0]);
  }
  const hourAverages = [...byHour.entries()]
    .map(([hour, values]) => ({
      hour,
      average: values.reduce((sum, value) => sum + value, 0) / values.length,
    }))
    .sort((left, right) => right.average - left.average);

  return {
    count: sessions.length,
    totalMinutes: Math.round(totalSeconds / 60),
    averagePeakRisk: Math.round(peaks.reduce((sum, value) => sum + value, 0) / peaks.length),
    alertsPerHour: totalSeconds
      ? Math.round((totalAlerts / (totalSeconds / 3600)) * 10) / 10
      : 0,
    worstHour: hourAverages[0]?.hour ?? null,
    safestHour: hourAverages[hourAverages.length - 1]?.hour ?? null,
  };
}
