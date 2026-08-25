/** Type surface for the feature bridge. Runtime lives in features.mjs. */

import type { Baseline, ClosureSample } from "./core.d.mts";

export type FeatureWindow = Record<string, number>;

export type TelemetrySample = ClosureSample & {
  ear?: number;
  mar?: number;
  yaw?: number;
  pitch?: number;
  gaze?: number;
  phoneVisible?: boolean;
  faceFound?: boolean;
};

export type LabeledWindow = {
  subjectId: string;
  label: string;
  features: FeatureWindow;
  extra?: Record<string, string | number>;
};

export declare const FEATURE_COLUMNS: string[];
export declare const LABEL_COLUMNS: string[];

export declare function buildFeatureWindow(
  samples: TelemetrySample[] | null | undefined,
  options?: {
    baseline?: Partial<Baseline>;
    now?: number | null;
    windowMs?: number;
    blinkTimes?: number[];
    yawnTimes?: number[];
  },
): FeatureWindow | null;

export declare function featureCsvHeader(extraColumns?: string[]): string;
export declare function featureCsvRow(
  features: FeatureWindow,
  options?: { subjectId?: string; label?: string; extra?: Record<string, string | number> },
): string;
export declare function buildFeatureCsv(rows: LabeledWindow[]): string;
