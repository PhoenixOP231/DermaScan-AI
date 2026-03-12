import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DermaScan AI — Skin Lesion Analysis",
  description:
    "AI-powered dermoscopy image classifier using ResNet18 with Grad-CAM explainability.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      {/* Light clinical theme — no dark class */}
      <body>{children}</body>
    </html>
  );
}
