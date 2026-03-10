import { NextRequest, NextResponse } from "next/server";

// Server-side – never exposed to the browser bundle.
// Set RAILWAY_API_URL (or NEXT_PUBLIC_API_URL) in Vercel environment variables.
const BACKEND =
  (process.env.RAILWAY_API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000")
    .replace(/\/$/, "");

export async function POST(req: NextRequest) {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return NextResponse.json({ detail: "Invalid form data." }, { status: 400 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND}/analyze`, {
      method: "POST",
      body: form,
    });
  } catch {
    return NextResponse.json(
      { detail: "Could not reach the backend server. Check RAILWAY_API_URL." },
      { status: 502 }
    );
  }

  const data = await upstream.json();
  return NextResponse.json(data, { status: upstream.status });
}
