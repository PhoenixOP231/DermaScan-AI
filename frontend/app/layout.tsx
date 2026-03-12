import type { Metadata } from "next";
import "./globals.css";
import ChatWidget from "./components/ChatWidget";

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
      <body>
        {children}
        <ChatWidget />
      </body>
    </html>
  );
}
