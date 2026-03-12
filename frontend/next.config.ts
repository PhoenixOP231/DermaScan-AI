import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow <img> tags to load base64 data-URIs from the inference response
  images: { unoptimized: true },

  // In local development, proxy /api/analyze to the FastAPI backend so you
  // can run `uvicorn backend.main:app` and use hot-reload without Vercel CLI.
  // In Vercel production, vercel.json routes /api/analyze to api/analyze.py
  // (the Python serverless function), so this rewrite is never applied there.
  async rewrites() {
    if (process.env.NODE_ENV === "development") {
      const backendUrl =
        process.env.NEXT_PUBLIC_API_URL ||
        process.env.RAILWAY_API_URL ||
        "http://localhost:8000";

      return [
        {
          source: "/api/analyze",
          destination: `${backendUrl}/analyze`,
        },
        {
          source: "/api/health",
          destination: `${backendUrl}/health`,
        },
      ];
    }
    return [];
  },
};

export default nextConfig;

