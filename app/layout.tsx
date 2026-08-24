import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const metadataBase = new URL(`${protocol}://${host}`);
  const description =
    "A free, privacy-first driver monitoring research system with on-device drowsiness, gaze, head pose, yawn, PERCLOS, and phone detection.";

  return {
    metadataBase,
    title: "Aegis Drive — Private Driver Drowsiness & Distraction Detection",
    description,
    applicationName: "Aegis Drive",
    keywords: [
      "driver drowsiness detection",
      "driver distraction detection",
      "on-device computer vision",
      "MediaPipe",
      "PERCLOS",
      "road safety",
    ],
    authors: [{ name: "Aegis Drive Project" }],
    creator: "Aegis Drive Project",
    openGraph: {
      title: "Aegis Drive — Guardian OS",
      description: "See fatigue before it becomes a decision. Private, local, and free.",
      type: "website",
      siteName: "Aegis Drive",
      images: [{ url: "/og.png", width: 1731, height: 909, alt: "Aegis Drive Guardian OS" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Aegis Drive — Guardian OS",
      description: "Private, on-device driver drowsiness and distraction detection.",
      images: ["/og.png"],
    },
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
