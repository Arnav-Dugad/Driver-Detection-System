import type { Metadata } from "next";
import ModelLab from "./ModelLab";
import "./lab.css";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Model Lab · Driver Drowsiness & Distraction Detection",
  description:
    "Held-out evaluation of the eye-state CNN and drowsiness GRU on UTA-RLDD, DMD, and MRL Eye, with a live side-by-side shadow mode.",
};

export default function LabPage() {
  return <ModelLab />;
}
