"use client";

import { useCallback, useRef, useState } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface AnalyzeResponse {
  is_skin_valid: boolean;
  predicted_class: string;
  risk_level: string;
  confidences: Record<string, number>; // { nv: 0.72, mel: 0.10, … }
  grad_cam_base64: string;
}

// Display labels & colours for each HAM10000 abbreviation
const CLASS_META: Record<
  string,
  { label: string; colour: string }
> = {
  nv:    { label: "Melanocytic Nevi",             colour: "#10b981" },
  mel:   { label: "Melanoma",                     colour: "#ef4444" },
  bkl:   { label: "Benign Keratosis-like",        colour: "#10b981" },
  bcc:   { label: "Basal Cell Carcinoma",         colour: "#ef4444" },
  akiec: { label: "Actinic Keratoses",            colour: "#f59e0b" },
  vasc:  { label: "Vascular Lesions",             colour: "#10b981" },
  df:    { label: "Dermatofibroma",               colour: "#10b981" },
};

const RISK_STYLE: Record<string, string> = {
  Benign:         "text-emerald-400 bg-emerald-400/10 border-emerald-400/30",
  Malignant:      "text-red-400     bg-red-400/10     border-red-400/30",
  "Pre-cancerous":"text-amber-400   bg-amber-400/10   border-amber-400/30",
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── Subcomponents ────────────────────────────────────────────────────────────

function GlowCard({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={
        "relative rounded-2xl border border-white/[0.08] bg-white/[0.03] " +
        "backdrop-blur-md shadow-[0_4px_32px_rgba(0,0,0,0.5)] " +
        "transition-all duration-300 hover:border-indigo-500/30 " +
        "hover:shadow-[0_4px_40px_rgba(99,102,241,0.12)] " +
        className
      }
    >
      {children}
    </div>
  );
}

function ConfidenceBar({
  abbr,
  value,
  delay,
}: {
  abbr: string;
  value: number;
  delay: number;
}) {
  const meta  = CLASS_META[abbr] ?? { label: abbr.toUpperCase(), colour: "#6366f1" };
  const pct   = (value * 100).toFixed(1);
  const width = `${(value * 100).toFixed(2)}%`;

  return (
    <div className="group flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-slate-300 tracking-wide">
          {meta.label}
        </span>
        <span
          className="font-mono font-semibold"
          style={{ color: meta.colour }}
        >
          {pct}%
        </span>
      </div>

      {/* Track */}
      <div className="h-1.5 w-full rounded-full bg-white/[0.06] overflow-hidden">
        {/* Fill */}
        <div
          className="animate-bar h-full rounded-full"
          style={
            {
              "--bar-width": width,
              background: `linear-gradient(90deg, ${meta.colour}99, ${meta.colour})`,
              animationDelay: `${delay}ms`,
            } as React.CSSProperties
          }
        />
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [preview,   setPreview]   = useState<string | null>(null);
  const [file,      setFile]      = useState<File | null>(null);
  const [loading,   setLoading]   = useState(false);
  const [result,    setResult]    = useState<AnalyzeResponse | null>(null);
  const [error,     setError]     = useState<string | null>(null);
  const [dragging,  setDragging]  = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);

  // ── image selection ──────────────────────────────────────────────────────
  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload a JPEG or PNG image.");
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  // ── drag & drop ──────────────────────────────────────────────────────────
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  };
  const onDragLeave = () => setDragging(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  // ── API call ─────────────────────────────────────────────────────────────
  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body:   form,
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(
          (detail as { detail?: string }).detail ??
            `Server error ${res.status}`
        );
      }

      const data: AnalyzeResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "An unexpected error occurred."
      );
    } finally {
      setLoading(false);
    }
  };

  // ── derived display data ─────────────────────────────────────────────────
  const sortedConfidences = result
    ? Object.entries(result.confidences).sort((a, b) => b[1] - a[1])
    : [];

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div className="relative z-10 min-h-dvh flex flex-col">
      {/* ══════════ Header ══════════ */}
      <header className="border-b border-white/[0.06] bg-[#020817]/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div
              className="h-8 w-8 rounded-lg flex items-center justify-center"
              style={{
                background:
                  "linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)",
              }}
            >
              <svg
                viewBox="0 0 24 24"
                className="h-4.5 w-4.5 text-white fill-current"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M9.5 2a.5.5 0 0 1 .5.5v1h4v-1a.5.5 0 0 1 1 0v1h1a2 2 0 0 1 2 2v1H6V5.5a2 2 0 0 1 2-2h1v-1a.5.5 0 0 1 .5-.5ZM4 8h16v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8Zm8 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0 1.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z" />
              </svg>
            </div>
            <span className="font-bold text-lg tracking-tight text-white">
              DermaScan{" "}
              <span
                className="font-light"
                style={{
                  background: "linear-gradient(90deg, #6366f1, #06b6d4)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                }}
              >
                AI
              </span>
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="hidden sm:inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Model Online
            </span>
          </div>
        </div>
      </header>

      {/* ══════════ Hero ══════════ */}
      <section className="mx-auto max-w-6xl w-full px-6 pt-14 pb-8 text-center">
        <div className="inline-flex items-center gap-2 rounded-full border border-indigo-500/20 bg-indigo-500/5 px-4 py-1.5 text-xs font-medium text-indigo-300 mb-6">
          <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 17.93V18a1 1 0 0 0-2 0v1.93A8 8 0 0 1 4.07 13H6a1 1 0 0 0 0-2H4.07A8 8 0 0 1 11 4.07V6a1 1 0 0 0 2 0V4.07A8 8 0 0 1 19.93 11H18a1 1 0 0 0 0 2h1.93A8 8 0 0 1 13 19.93z" />
          </svg>
          HAM10000 · ResNet18 · Grad-CAM
        </div>

        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-white mb-4 leading-tight">
          Skin Lesion{" "}
          <span
            style={{
              background: "linear-gradient(90deg, #6366f1, #06b6d4)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Intelligence
          </span>
        </h1>
        <p className="max-w-xl mx-auto text-slate-400 text-base leading-relaxed">
          Upload a dermoscopy image to receive an instant AI-powered
          classification with visual explainability via Grad-CAM heatmaps.
        </p>
      </section>

      {/* ══════════ Main content ══════════ */}
      <main className="mx-auto max-w-6xl w-full px-6 pb-20 flex-1">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ── Left column: upload + button ── */}
          <div className="flex flex-col gap-5">
            <GlowCard>
              <div className="p-5">
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-4">
                  Upload Image
                </p>

                {/* Drop zone */}
                <div
                  role="button"
                  tabIndex={0}
                  aria-label="Upload image"
                  className={
                    "relative flex flex-col items-center justify-center gap-3 " +
                    "rounded-xl border-2 border-dashed transition-all duration-200 " +
                    "cursor-pointer min-h-[220px] select-none " +
                    (dragging
                      ? "border-indigo-400 bg-indigo-500/10"
                      : preview
                      ? "border-indigo-500/40 bg-transparent"
                      : "border-white/[0.1] bg-white/[0.02] hover:border-indigo-500/40 hover:bg-indigo-500/5")
                  }
                  onClick={() => inputRef.current?.click()}
                  onKeyDown={(e) =>
                    (e.key === "Enter" || e.key === " ") &&
                    inputRef.current?.click()
                  }
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                >
                  {preview ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={preview}
                      alt="Uploaded dermoscopy image"
                      className="rounded-lg object-contain max-h-52 w-full"
                    />
                  ) : (
                    <>
                      <div className="h-12 w-12 rounded-full bg-white/[0.04] border border-white/[0.08] flex items-center justify-center">
                        <svg
                          className="h-6 w-6 text-slate-500"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
                          />
                        </svg>
                      </div>
                      <div className="text-center">
                        <p className="text-sm font-medium text-slate-300">
                          Drop image here
                        </p>
                        <p className="text-xs text-slate-500 mt-1">
                          or click to browse · JPEG, PNG, BMP
                        </p>
                      </div>
                    </>
                  )}

                  {/* Re-upload overlay on hover when preview exists */}
                  {preview && (
                    <div className="absolute inset-0 rounded-xl bg-black/60 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center gap-2 text-sm font-medium text-white">
                      <svg
                        className="h-4 w-4"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth={2}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99"
                        />
                      </svg>
                      Replace image
                    </div>
                  )}
                </div>

                <input
                  ref={inputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/bmp,image/webp"
                  className="hidden"
                  onChange={onInputChange}
                />
              </div>
            </GlowCard>

            {/* Analyse button */}
            <button
              onClick={analyze}
              disabled={!file || loading}
              className={
                "w-full rounded-xl py-3.5 text-sm font-semibold tracking-wide " +
                "transition-all duration-200 relative overflow-hidden " +
                (!file || loading
                  ? "bg-white/[0.04] text-slate-500 cursor-not-allowed border border-white/[0.06]"
                  : "text-white border border-transparent cursor-pointer " +
                    "hover:shadow-[0_0_24px_rgba(99,102,241,0.5)] active:scale-[0.98]")
              }
              style={
                file && !loading
                  ? {
                      background:
                        "linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)",
                    }
                  : {}
              }
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg
                    className="h-4 w-4 animate-spin"
                    viewBox="0 0 24 24"
                    fill="none"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z"
                    />
                  </svg>
                  Analysing…
                </span>
              ) : (
                "Run Analysis"
              )}
            </button>

            {/* Error banner */}
            {error && (
              <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
                <span className="font-semibold">Error — </span>
                {error}
              </div>
            )}

            {/* Skin validation rejection */}
            {result && !result.is_skin_valid && (
              <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
                <p className="font-semibold mb-1">Not a skin image</p>
                <p className="text-amber-400/70 text-xs leading-relaxed">
                  The image did not pass the dual color-space skin-pixel
                  validation (Kovac RGB ∩ Peer YCbCr). Please upload a
                  dermoscopy photograph of a skin lesion.
                </p>
              </div>
            )}
          </div>

          {/* ── Right column: results ── */}
          <div className="flex flex-col gap-5">
            {result && result.is_skin_valid ? (
              <>
                {/* Prediction card */}
                <GlowCard>
                  <div className="p-5">
                    <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-4">
                      Prediction
                    </p>
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <h2 className="text-xl font-bold text-white leading-tight">
                          {result.predicted_class}
                        </h2>
                        <p className="text-slate-400 text-sm mt-1">
                          Top-1 classification
                        </p>
                      </div>
                      <span
                        className={
                          "mt-0.5 shrink-0 rounded-full border px-3 py-1 text-xs font-semibold " +
                          (RISK_STYLE[result.risk_level] ?? "text-slate-300 bg-slate-700 border-slate-600")
                        }
                      >
                        {result.risk_level}
                      </span>
                    </div>

                    {/* Top confidence */}
                    <div className="mt-4 rounded-lg bg-white/[0.03] border border-white/[0.06] px-4 py-3">
                      <p className="text-xs text-slate-500 mb-1">
                        Top confidence
                      </p>
                      <p className="text-2xl font-extrabold text-white font-mono">
                        {(
                          (result.confidences[
                            Object.entries(result.confidences).sort(
                              (a, b) => b[1] - a[1]
                            )[0][0]
                          ] ?? 0) * 100
                        ).toFixed(1)}
                        <span className="text-lg text-slate-400 font-normal">
                          %
                        </span>
                      </p>
                    </div>
                  </div>
                </GlowCard>

                {/* Confidence bars */}
                <GlowCard>
                  <div className="p-5">
                    <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-4">
                      All Class Probabilities
                    </p>
                    <div className="flex flex-col gap-3.5">
                      {sortedConfidences.map(([abbr, val], i) => (
                        <ConfidenceBar
                          key={abbr}
                          abbr={abbr}
                          value={val}
                          delay={i * 60}
                        />
                      ))}
                    </div>
                  </div>
                </GlowCard>
              </>
            ) : (
              /* Placeholder when no results yet */
              <GlowCard className="flex-1">
                <div className="p-5 h-full min-h-[260px] flex flex-col items-center justify-center gap-3 text-center">
                  <div className="h-12 w-12 rounded-full bg-white/[0.03] border border-white/[0.08] flex items-center justify-center">
                    <svg
                      className="h-6 w-6 text-slate-600"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M3.75 3v11.25A2.25 2.25 0 0 0 6 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0 1 18 16.5h-2.25m-7.5 0h7.5m-7.5 0-1 3m8.5-3 1 3m0 0 .5 1.5m-.5-1.5h-9.5m0 0-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6"
                      />
                    </svg>
                  </div>
                  <p className="text-slate-500 text-sm">
                    Results will appear here
                  </p>
                </div>
              </GlowCard>
            )}
          </div>
        </div>

        {/* ── Grad-CAM + image comparison (full width) ── */}
        {result && result.is_skin_valid && result.grad_cam_base64 && (
          <div className="mt-6">
            <GlowCard>
              <div className="p-5">
                <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-4">
                  Visual Explanation — Grad-CAM Heatmap
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Original */}
                  <div className="flex flex-col gap-2">
                    <p className="text-xs text-slate-500 font-medium">
                      Original Image
                    </p>
                    {preview && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={preview}
                        alt="Original uploaded image"
                        className="w-full rounded-xl object-cover aspect-square border border-white/[0.06]"
                      />
                    )}
                  </div>

                  {/* Heatmap */}
                  <div className="flex flex-col gap-2">
                    <p className="text-xs text-slate-500 font-medium">
                      Grad-CAM Activation Heatmap
                    </p>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${result.grad_cam_base64}`}
                      alt="Grad-CAM heatmap overlay"
                      className="w-full rounded-xl object-cover aspect-square border border-white/[0.06]"
                    />
                  </div>
                </div>

                {/* Legend */}
                <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-500">
                  <span className="font-medium text-slate-400">Heatmap key:</span>
                  {[
                    { label: "High activation", colour: "#ef4444" },
                    { label: "Medium",           colour: "#f59e0b" },
                    { label: "Low activation",   colour: "#3b82f6" },
                  ].map(({ label, colour }) => (
                    <span key={label} className="flex items-center gap-1.5">
                      <span
                        className="h-2.5 w-2.5 rounded-full"
                        style={{ background: colour }}
                      />
                      {label}
                    </span>
                  ))}
                </div>
              </div>
            </GlowCard>
          </div>
        )}

        {/* ── Disclaimer ── */}
        <div className="mt-6 rounded-xl border border-amber-500/20 bg-amber-500/5 px-5 py-4">
          <p className="text-xs text-amber-400/80 leading-relaxed">
            <span className="font-semibold text-amber-400">
              Medical Disclaimer —{" "}
            </span>
            This tool is intended for research and educational purposes only.
            It does not constitute medical advice or replace professional
            dermatological diagnosis. Always consult a qualified healthcare
            provider for clinical decisions.
          </p>
        </div>
      </main>

      {/* ══════════ Footer ══════════ */}
      <footer className="border-t border-white/[0.06] bg-[#020817]/60 backdrop-blur-md">
        <div className="mx-auto max-w-6xl px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div
              className="h-6 w-6 rounded-md flex items-center justify-center"
              style={{
                background:
                  "linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)",
              }}
            >
              <svg
                viewBox="0 0 24 24"
                className="h-3.5 w-3.5 text-white fill-current"
              >
                <path d="M9.5 2a.5.5 0 0 1 .5.5v1h4v-1a.5.5 0 0 1 1 0v1h1a2 2 0 0 1 2 2v1H6V5.5a2 2 0 0 1 2-2h1v-1a.5.5 0 0 1 .5-.5ZM4 8h16v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8Zm8 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0 1.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z" />
              </svg>
            </div>
            <span className="text-sm font-semibold text-slate-300">
              DermaScan AI Labs
            </span>
            <span className="hidden sm:inline text-slate-600">·</span>
            <span className="hidden sm:inline text-sm text-slate-500">
              Research &amp; Development
            </span>
          </div>

          <div className="flex items-center gap-5 text-xs text-slate-600">
            <span>HAM10000 Dataset</span>
            <span>·</span>
            <span>ResNet18 Architecture</span>
            <span>·</span>
            <span>Grad-CAM Explainability</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
