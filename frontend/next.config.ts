import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow <img> tags to load base64 data-URIs from the FastAPI response
  images: { unoptimized: true },
};

export default nextConfig;
