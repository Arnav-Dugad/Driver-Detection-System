import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const githubPagesUrl = "https://arnav-dugad.github.io/Driver-Detection-System/";
const deployedUrl =
  process.env.NEXT_PUBLIC_SITE_URL ||
  "https://driver-detection-arnav-dugad.arnavd.chatgpt.site/";
const metadataBase = new URL(
  process.env.GITHUB_PAGES === "true" ? githubPagesUrl : deployedUrl,
);
const description =
  "A free, privacy-first driver monitoring research system with on-device drowsiness, gaze, head pose, yawn, PERCLOS, phone detection, and multilingual voice warnings.";
const socialImage = new URL("og.png", metadataBase).toString();
const favicon = new URL("favicon.svg", metadataBase).toString();

export const metadata: Metadata = {
  metadataBase,
  title: "Driver Drowsiness & Distraction Detection System",
  description,
  applicationName: "Driver Drowsiness & Distraction Detection System",
  keywords: [
    "driver drowsiness detection",
    "driver distraction detection",
    "on-device computer vision",
    "MediaPipe",
    "PERCLOS",
    "multilingual driver alerts",
    "road safety",
  ],
  authors: [{ name: "Driver Drowsiness & Distraction Detection System Project" }],
  creator: "Driver Drowsiness & Distraction Detection System Project",
  openGraph: {
    title: "Driver Drowsiness & Distraction Detection System",
    description: "Private, on-device driver monitoring with multilingual warnings.",
    type: "website",
    siteName: "Driver Drowsiness & Distraction Detection System",
    images: [{ url: socialImage, width: 1731, height: 909, alt: "Driver Drowsiness and Distraction Detection System" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Driver Drowsiness & Distraction Detection System",
    description: "Private, on-device driver drowsiness and distraction detection.",
    images: [socialImage],
  },
  icons: {
    icon: favicon,
    shortcut: favicon,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
