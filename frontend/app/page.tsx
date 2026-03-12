"use client";

import { useCallback, useRef, useState } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface AnalyzeResponse {
  is_skin_valid:   boolean;
  predicted_class: string;
  risk_level:      string;
  confidences:     Record<string, number>;
  grad_cam_base64: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const CLASS_META: Record<string, { label: string; colour: string; track: string }> = {
  nv:    { label: "Melanocytic Nevi",            colour: "#16a34a", track: "#dcfce7" },
  mel:   { label: "Melanoma",                    colour: "#dc2626", track: "#fee2e2" },
  bkl:   { label: "Benign Keratosis-like",       colour: "#16a34a", track: "#dcfce7" },
  bcc:   { label: "Basal Cell Carcinoma",        colour: "#dc2626", track: "#fee2e2" },
  akiec: { label: "Actinic Keratoses",           colour: "#b45309", track: "#fef3c7" },
  vasc:  { label: "Vascular Lesions",            colour: "#16a34a", track: "#dcfce7" },
  df:    { label: "Dermatofibroma",              colour: "#16a34a", track: "#dcfce7" },
};

const RISK_BADGE: Record<string, string> = {
  Benign:          "badge-benign",
  Malignant:       "badge-malignant",
  "Pre-cancerous": "badge-precancerous",
};

// ─── Sidebar nav items ────────────────────────────────────────────────────────

const NAV = [
  {
    label: "Analysis",
    icon: (
      <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2m0 10a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2M9 7a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2m0 10V7m0 10a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2a2 2 0 0 0-2 2" />
      </svg>
    ),
  },
  {
    label: "Model Info",
    icon: (
      <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M11.42 15.17 17.25 21A2.652 2.652 0 0 0 21 17.25l-5.877-5.877M11.42 15.17l2.496-3.03c.317-.384.74-.626 1.208-.766M11.42 15.17l-4.655 5.653a2.548 2.548 0 1 1-3.586-3.586l6.837-5.63m5.108-.233c.55-.164 1.163-.188 1.743-.14a4.5 4.5 0 0 0 4.486-6.336l-3.276 3.277a3.004 3.004 0 0 1-2.25-2.25l3.276-3.276a4.5 4.5 0 0 0-6.336 4.486c.091 1.076-.071 2.264-.904 2.95l-.102.085m-1.745 1.437L5.909 7.5H4.5L2.25 3.75l1.5-1.5L7.5 4.5v1.409l4.26 4.26m-1.745 1.437 1.745-1.437m6.615 8.206L15.75 15.75M4.867 19.125h.008v.008h-.008v-.008Z" />
      </svg>
    ),
  },
  {
    label: "About",
    icon: (
      <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z" />
      </svg>
    ),
  },
];

// ─── Sub-components ───────────────────────────────────────────────────────────

/** White bordered clinical card with optional titled header */
function Card({
  title,
  badge,
  children,
  className = "",
}: {
  title?: string;
  badge?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`ds-card ${className}`}>
      {title && (
        <div className="ds-card-header">
          <span className="ds-label">{title}</span>
          {badge}
        </div>
      )}
      {children}
    </div>
  );
}

/** SVG circular progress ring with animated stroke */
function RingChart({
  value,
  colour,
  track,
  label,
  size = 80,
}: {
  value: number;
  colour: string;
  track: string;
  label: string;
  size?: number;
}) {
  const stroke = 7;
  const r      = (size - stroke) / 2;
  const circ   = 2 * Math.PI * r;
  const offset = circ * (1 - value);
  const pct    = (value * 100).toFixed(1);

  return (
    <div className="flex flex-col items-center gap-1.5">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ transform: "rotate(-90deg)" }}
        aria-label={`${label}: ${pct}%`}
      >
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={track} strokeWidth={stroke} />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke={colour}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circ}
          className="animate-ring"
          style={
            {
              "--circ":   circ,
              "--offset": offset,
              strokeDashoffset: circ,
            } as React.CSSProperties
          }
        />
      </svg>
      <span className="text-xs font-semibold tabular-nums" style={{ color: colour }}>
        {pct}%
      </span>
      <span
        className="text-[10px] text-center leading-tight"
        style={{ color: "var(--text-muted)", maxWidth: size }}
      >
        {label}
      </span>
    </div>
  );
}

/** Horizontal probability bar row */
function ConfBar({ abbr, value, delay }: { abbr: string; value: number; delay: number }) {
  const meta  = CLASS_META[abbr] ?? { label: abbr.toUpperCase(), colour: "#2563eb", track: "#dbeafe" };
  const pct   = (value * 100).toFixed(1);
  const width = `${(value * 100).toFixed(2)}%`;

  return (
    <div className="flex items-center gap-3">
      <span
        className="w-36 shrink-0 text-xs font-medium truncate"
        style={{ color: "var(--text-secondary)" }}
      >
        {meta.label}
      </span>
      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: meta.track }}>
        <div
          className="h-full rounded-full animate-bar"
          style={
            {
              "--bar-width": width,
              background: meta.colour,
              animationDelay: `${delay}ms`,
            } as React.CSSProperties
          }
        />
      </div>
      <span
        className="w-10 shrink-0 text-right text-xs font-semibold tabular-nums"
        style={{ color: meta.colour }}
      >
        {pct}%
      </span>
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
  const [activeNav, setActiveNav] = useState("Analysis");

  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload a JPEG, PNG, or BMP image.");
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

  const onDragOver  = (e: React.DragEvent) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);
  const onDrop      = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error((detail as { detail?: string }).detail ?? `Server error ${res.status}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const sorted  = result ? Object.entries(result.confidences).sort((a, b) => b[1] - a[1]) : [];
  const top3    = sorted.slice(0, 3);
  const topConf = sorted[0]?.[1] ?? 0;

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg-canvas)" }}>

      {/* ════════════════════════════════ SIDEBAR ════════════════════════════ */}
      <aside
        className="hidden lg:flex flex-col w-56 shrink-0 h-full overflow-y-auto"
        style={{ background: "var(--bg-surface)", borderRight: "1px solid var(--border)" }}
      >
        {/* Wordmark */}
        <div
          className="flex items-center gap-2.5 px-5 py-5"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div
            className="h-7 w-7 rounded-lg flex items-center justify-center shrink-0"
            style={{ background: "#2563eb" }}
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4 fill-white">
              <path d="M9.5 2a.5.5 0 0 1 .5.5v1h4v-1a.5.5 0 0 1 1 0v1h1a2 2 0 0 1 2 2v1H6V5.5a2 2 0 0 1 2-2h1v-1a.5.5 0 0 1 .5-.5ZM4 8h16v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8Zm8 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0 1.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-bold leading-none" style={{ color: "var(--text-primary)" }}>
              DermaScan
            </p>
            <p className="text-[10px] mt-0.5 font-semibold" style={{ color: "#2563eb" }}>
              AI Platform
            </p>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex flex-col gap-0.5 px-3 pt-4">
          <p className="ds-label px-2 pb-2">Navigation</p>
          {NAV.map((item) => (
            <button
              key={item.label}
              className={`nav-item ${activeNav === item.label ? "active" : ""}`}
              onClick={() => setActiveNav(item.label)}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>

        {/* Model status chip */}
        <div className="mt-auto px-4 pb-5">
          <div
            className="rounded-lg px-3 py-2.5 flex items-center gap-2"
            style={{ background: "var(--bg-canvas)", border: "1px solid var(--border)" }}
          >
            <span className="h-2 w-2 rounded-full bg-emerald-500 shrink-0 animate-pulse" />
            <div>
              <p className="text-[11px] font-semibold" style={{ color: "var(--text-primary)" }}>
                ResNet18 · ONNX
              </p>
              <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                HAM10000 · 7 classes
              </p>
            </div>
          </div>
        </div>
      </aside>

      {/* ══════════════════════════════ MAIN AREA ════════════════════════════ */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">

        {/* Top bar */}
        <header
          className="shrink-0 flex items-center justify-between px-4 lg:px-6 py-3.5"
          style={{ background: "var(--bg-surface)", borderBottom: "1px solid var(--border)" }}
        >
          <div className="flex items-center gap-3">
            {/* Mobile logo mark */}
            <div
              className="h-7 w-7 rounded-lg flex items-center justify-center shrink-0 lg:hidden"
              style={{ background: "#2563eb" }}
            >
              <svg viewBox="0 0 24 24" className="h-4 w-4 fill-white">
                <path d="M9.5 2a.5.5 0 0 1 .5.5v1h4v-1a.5.5 0 0 1 1 0v1h1a2 2 0 0 1 2 2v1H6V5.5a2 2 0 0 1 2-2h1v-1a.5.5 0 0 1 .5-.5ZM4 8h16v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8Zm8 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0 1.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z" />
              </svg>
            </div>
            <div>
              <h1 className="text-base font-bold" style={{ color: "var(--text-primary)" }}>
                {activeNav === "Analysis" ? "Lesion Analysis" : activeNav}
              </h1>
              <p className="text-xs hidden sm:block" style={{ color: "var(--text-muted)" }}>
                {activeNav === "Analysis"
                  ? "Upload a dermoscopy image for AI-powered classification"
                  : activeNav === "Model Info"
                  ? "Architecture, training details and performance metrics"
                  : "About this project and the research team"}
              </p>
            </div>
          </div>

          <span
            className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium"
            style={{ background: "#f0fdf4", color: "#16a34a", border: "1px solid #bbf7d0" }}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="hidden sm:inline">Model </span>Online
          </span>
        </header>

        {/* Scrollable content — extra bottom padding on mobile for the tab bar */}
        <main className="flex-1 overflow-y-auto p-5 pb-24 lg:pb-6 lg:p-6">
          <div className="max-w-5xl mx-auto flex flex-col gap-5">

            {/* ── Model Info page ── */}
            {activeNav === "Model Info" && (
              <>
                <Card title="Model Architecture">
                  <div className="p-5 flex flex-col gap-4">
                    {[
                      { label: "Base Architecture",  value: "ResNet18 (He et al., 2016)" },
                      { label: "Modification",        value: "Final FC layer replaced — 512 → 7 classes" },
                      { label: "Input Resolution",   value: "224 × 224 pixels (ImageNet standard)" },
                      { label: "Normalisation",       value: "Mean [0.485, 0.456, 0.406] · Std [0.229, 0.224, 0.225]" },
                      { label: "Inference Runtime",  value: "ONNX Runtime (CPU) — no PyTorch at inference" },
                      { label: "Model Size",          value: "~44 MB (ONNX)" },
                    ].map(({ label, value }) => (
                      <div key={label} className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-4">
                        <span className="w-48 shrink-0 ds-label">{label}</span>
                        <span className="text-sm" style={{ color: "var(--text-primary)" }}>{value}</span>
                      </div>
                    ))}
                  </div>
                </Card>

                <Card title="Training Details">
                  <div className="p-5 flex flex-col gap-4">
                    {[
                      { label: "Dataset",           value: "HAM10000 — Human Against Machine with 10,000 training images (ISIC 2018)" },
                      { label: "Total Images",      value: "10,015 dermoscopy images across 7 classes" },
                      { label: "Training Strategy", value: "Transfer learning — ImageNet pre-trained weights, fine-tuned on HAM10000" },
                      { label: "Optimiser",         value: "Adam · learning rate 1 × 10⁻⁴" },
                      { label: "Loss Function",     value: "Cross-Entropy Loss" },
                      { label: "Explainability",    value: "Class Activation Mapping (CAM) via layer4 feature maps" },
                    ].map(({ label, value }) => (
                      <div key={label} className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-4">
                        <span className="w-48 shrink-0 ds-label">{label}</span>
                        <span className="text-sm" style={{ color: "var(--text-primary)" }}>{value}</span>
                      </div>
                    ))}
                  </div>
                </Card>

                <Card title="Lesion Classes — HAM10000">
                  <div className="p-5">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {Object.entries(CLASS_META).map(([abbr, meta]) => (
                        <div
                          key={abbr}
                          className="flex items-center gap-3 rounded-lg px-3 py-2.5"
                          style={{ background: "var(--bg-canvas)", border: "1px solid var(--border)" }}
                        >
                          <span
                            className="h-2.5 w-2.5 rounded-full shrink-0"
                            style={{ background: meta.colour }}
                          />
                          <div>
                            <p className="text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
                              {meta.label}
                            </p>
                            <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                              {abbr.toUpperCase()}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>
              </>
            )}

            {/* ── About page ── */}
            {activeNav === "About" && (
              <>
                <Card title="About DermaScan AI">
                  <div className="p-5 flex flex-col gap-3">
                    <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                      DermaScan AI is an AI-powered dermoscopy image analysis tool that classifies skin
                      lesions into seven diagnostic categories using a fine-tuned ResNet18 convolutional
                      neural network. The tool provides real-time classification with visual explanations
                      via Grad-CAM activation heatmaps.
                    </p>
                    <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                      The model was trained on the publicly available HAM10000 dataset (ISIC 2018
                      Challenge), which contains 10,015 dermoscopic images collected from different
                      populations spanning over 20 years. It is deployed as a serverless Next.js
                      application using ONNX Runtime for efficient, dependency-light inference.
                    </p>
                  </div>
                </Card>

                <Card title="Technology Stack">
                  <div className="p-5">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {[
                        { label: "Frontend",     value: "Next.js 16, React 19, TypeScript" },
                        { label: "Styling",      value: "Tailwind CSS 3" },
                        { label: "ML Model",     value: "ResNet18 (ONNX)" },
                        { label: "Inference",    value: "ONNX Runtime (serverless)" },
                        { label: "Backend",      value: "Python · Flask · Vercel" },
                        { label: "Dataset",      value: "HAM10000 · ISIC 2018" },
                      ].map(({ label, value }) => (
                        <div
                          key={label}
                          className="rounded-lg px-3 py-2.5"
                          style={{ background: "var(--bg-canvas)", border: "1px solid var(--border)" }}
                        >
                          <p className="ds-label mb-0.5">{label}</p>
                          <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>{value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>

                <div
                  className="rounded-xl px-5 py-4"
                  style={{ background: "var(--amber-bg)", border: "1px solid var(--amber-bdr)" }}
                >
                  <p className="text-xs leading-relaxed" style={{ color: "#92400e" }}>
                    <span className="font-semibold" style={{ color: "var(--amber)" }}>Medical Disclaimer — </span>
                    This tool is for research and educational purposes only. It does not constitute medical
                    advice or replace professional dermatological diagnosis. Always consult a qualified
                    healthcare provider for any skin concern.
                  </p>
                </div>
              </>
            )}

            {/* ── Row 1: Upload + Prediction summary ── */}
            {activeNav === "Analysis" && (<>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

              {/* Upload card */}
              <Card title="Image Upload">
                <div className="p-4 flex flex-col gap-4">

                  {/* Drop zone */}
                  <div
                    role="button"
                    tabIndex={0}
                    aria-label="Upload dermoscopy image"
                    className="relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed cursor-pointer min-h-[200px] select-none transition-all duration-150"
                    style={{
                      borderColor: dragging ? "#2563eb" : preview ? "#bfdbfe" : "#d1d9e0",
                      background:  dragging ? "#eff6ff" : preview ? "transparent" : "#fafbfc",
                    }}
                    onClick={() => inputRef.current?.click()}
                    onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
                    onDragOver={onDragOver}
                    onDragLeave={onDragLeave}
                    onDrop={onDrop}
                  >
                    {preview ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={preview}
                        alt="Uploaded dermoscopy image"
                        className="rounded-lg object-contain max-h-48 w-full"
                      />
                    ) : (
                      <>
                        <div
                          className="h-11 w-11 rounded-full flex items-center justify-center"
                          style={{ background: "#eff6ff", border: "1px solid #bfdbfe" }}
                        >
                          <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
                          </svg>
                        </div>
                        <div className="text-center">
                          <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                            Drop image here
                          </p>
                          <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                            or click to browse · JPEG, PNG, BMP
                          </p>
                        </div>
                      </>
                    )}
                    {preview && (
                      <div className="absolute inset-0 rounded-xl bg-black/40 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center gap-2 text-sm font-medium text-white">
                        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
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

                  {/* Analyse button */}
                  <button
                    onClick={analyze}
                    disabled={!file || loading}
                    className="w-full rounded-lg py-2.5 text-sm font-semibold transition-all duration-150 active:scale-[0.98]"
                    style={
                      !file || loading
                        ? { background: "var(--bg-canvas)", color: "var(--text-muted)", border: "1px solid var(--border)", cursor: "not-allowed" }
                        : { background: "#2563eb", color: "#ffffff", border: "none", cursor: "pointer", boxShadow: "0 1px 4px rgba(37,99,235,0.35)" }
                    }
                  >
                    {loading ? (
                      <span className="flex items-center justify-center gap-2">
                        <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
                        </svg>
                        Analysing…
                      </span>
                    ) : (
                      "Run Analysis"
                    )}
                  </button>

                  {/* Error */}
                  {error && (
                    <div
                      className="rounded-lg px-4 py-3 text-xs"
                      style={{ background: "#fef2f2", color: "#dc2626", border: "1px solid #fecaca" }}
                    >
                      <span className="font-semibold">Error — </span>{error}
                    </div>
                  )}

                  {/* Skin validation failed */}
                  {result && !result.is_skin_valid && (
                    <div
                      className="rounded-lg px-4 py-3 text-xs"
                      style={{ background: "#fffbeb", color: "#b45309", border: "1px solid #fde68a" }}
                    >
                      <p className="font-semibold mb-0.5">Skin validation failed</p>
                      <p style={{ color: "#92400e" }}>
                        The image did not pass the dual colour-space skin-pixel gate (Kovac RGB ∩ Peer YCbCr).
                        Please upload a dermoscopy photograph of a skin lesion.
                      </p>
                    </div>
                  )}
                </div>
              </Card>

              {/* Diagnosis summary card */}
              {result && result.is_skin_valid ? (
                <Card title="Diagnosis Summary">
                  <div className="p-5 flex flex-col gap-5">

                    {/* Class name + risk badge */}
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-lg font-bold leading-snug" style={{ color: "var(--text-primary)" }}>
                          {result.predicted_class}
                        </p>
                        <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                          Top-1 ResNet18 classification
                        </p>
                      </div>
                      <span
                        className={`shrink-0 mt-0.5 px-3 py-1 rounded-full text-xs font-semibold ${RISK_BADGE[result.risk_level] ?? "badge-benign"}`}
                      >
                        {result.risk_level}
                      </span>
                    </div>

                    <hr style={{ borderColor: "var(--border)" }} />

                    {/* Top-3 ring charts */}
                    <div>
                      <p className="ds-label mb-4">Top-3 Class Probabilities</p>
                      <div className="flex items-start justify-around gap-2">
                        {top3.map(([abbr, val]) => {
                          const m = CLASS_META[abbr] ?? { label: abbr, colour: "#2563eb", track: "#dbeafe" };
                          return (
                            <RingChart
                              key={abbr}
                              value={val}
                              colour={m.colour}
                              track={m.track}
                              label={m.label}
                              size={76}
                            />
                          );
                        })}
                      </div>
                    </div>

                    <hr style={{ borderColor: "var(--border)" }} />

                    {/* Metric grid */}
                    <div className="grid grid-cols-2 gap-3">
                      {[
                        { label: "Architecture", value: "ResNet18"                        },
                        { label: "Dataset",      value: "HAM10000"                        },
                        { label: "Confidence",   value: `${(topConf * 100).toFixed(1)}%` },
                        { label: "Classes",      value: "7"                               },
                      ].map(({ label, value }) => (
                        <div
                          key={label}
                          className="rounded-lg px-3 py-2.5"
                          style={{ background: "var(--bg-canvas)", border: "1px solid var(--border)" }}
                        >
                          <p
                            className="text-[10px] font-semibold uppercase tracking-wide"
                            style={{ color: "var(--text-muted)" }}
                          >
                            {label}
                          </p>
                          <p className="text-sm font-bold mt-0.5" style={{ color: "var(--text-primary)" }}>
                            {value}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>
              ) : (
                /* Empty state */
                <Card title="Diagnosis Summary">
                  <div className="p-5 min-h-[280px] flex flex-col items-center justify-center gap-3 text-center">
                    <div
                      className="h-12 w-12 rounded-full flex items-center justify-center"
                      style={{ background: "var(--bg-canvas)", border: "1px solid var(--border)" }}
                    >
                      <svg className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="#8a96a8" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 0 0 6 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0 1 18 16.5h-2.25m-7.5 0h7.5m-7.5 0-1 3m8.5-3 1 3m0 0 .5 1.5m-.5-1.5h-9.5m0 0-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
                      </svg>
                    </div>
                    <p className="text-sm font-medium" style={{ color: "var(--text-muted)" }}>
                      Upload an image and click Run Analysis
                    </p>
                    <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                      Results will appear here
                    </p>
                  </div>
                </Card>
              )}
            </div>

            {/* ── Row 2: All probabilities (horizontal bars) ── */}
            {result && result.is_skin_valid && (
              <Card title="All Class Probabilities">
                <div className="p-5 flex flex-col gap-3">
                  {sorted.map(([abbr, val], i) => (
                    <ConfBar key={abbr} abbr={abbr} value={val} delay={i * 55} />
                  ))}
                </div>
              </Card>
            )}

            {/* ── Row 3: Grad-CAM heatmap side-by-side ── */}
            {result && result.is_skin_valid && result.grad_cam_base64 && (
              <Card
                title="Visual Explanation — Activation Heatmap"
                badge={
                  <span
                    className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold"
                    style={{ background: "#eff6ff", color: "#2563eb", border: "1px solid #bfdbfe" }}
                  >
                    CAM / Grad-CAM
                  </span>
                }
              >
                <div className="p-5">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="flex flex-col gap-2">
                      <p className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>
                        Original Image
                      </p>
                      {preview && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={preview}
                          alt="Original uploaded image"
                          className="w-full rounded-xl object-cover aspect-square"
                          style={{ border: "1px solid var(--border)" }}
                        />
                      )}
                    </div>
                    <div className="flex flex-col gap-2">
                      <p className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>
                        Activation Heatmap
                      </p>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`data:image/png;base64,${result.grad_cam_base64}`}
                        alt="Grad-CAM activation heatmap"
                        className="w-full rounded-xl object-cover aspect-square"
                        style={{ border: "1px solid var(--border)" }}
                      />
                    </div>
                  </div>
                  {/* Legend */}
                  <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2">
                    <span className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>
                      Heatmap key:
                    </span>
                    {[
                      { label: "High activation", colour: "#ef4444" },
                      { label: "Medium",           colour: "#f59e0b" },
                      { label: "Low activation",   colour: "#3b82f6" },
                    ].map(({ label, colour }) => (
                      <span key={label} className="flex items-center gap-1.5">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ background: colour }} />
                        <span className="text-xs" style={{ color: "var(--text-muted)" }}>{label}</span>
                      </span>
                    ))}
                  </div>
                </div>
              </Card>
            )}

            {/* ── Medical disclaimer ── */}
            <div
              className="rounded-xl px-5 py-4"
              style={{ background: "#fffbeb", border: "1px solid #fde68a" }}
            >
              <p className="text-xs leading-relaxed" style={{ color: "#92400e" }}>
                <span className="font-semibold" style={{ color: "#b45309" }}>
                  Medical Disclaimer —{" "}
                </span>
                This tool is intended for research and educational purposes only. It does not constitute
                medical advice or replace professional dermatological diagnosis. Always consult a qualified
                healthcare provider for clinical decisions.
              </p>
            </div>
            </>)}

          </div>
        </main>

        {/* Footer */}
        <footer
          className="shrink-0 px-6 py-3 hidden lg:flex items-center justify-between"
          style={{ borderTop: "1px solid var(--border)", background: "var(--bg-surface)" }}
        >
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>
            DermaScan AI
          </p>
          <p className="text-xs hidden sm:block" style={{ color: "var(--text-muted)" }}>
            HAM10000 · ResNet18 · Grad-CAM
          </p>
        </footer>

        {/* ══════════════ MOBILE BOTTOM TAB BAR (hidden on lg+) ══════════════ */}
        <nav
          className="lg:hidden fixed bottom-0 left-0 right-0 z-40 flex"
          style={{
            background  : "var(--bg-surface)",
            borderTop   : "1px solid var(--border)",
            boxShadow   : "0 -2px 12px rgba(15,25,35,0.08)",
            paddingBottom: "env(safe-area-inset-bottom, 0px)",
          }}
        >
          {NAV.map((item) => {
            const active = activeNav === item.label;
            return (
              <button
                key={item.label}
                onClick={() => setActiveNav(item.label)}
                className="relative flex-1 flex flex-col items-center justify-center gap-1 py-2.5 transition-colors duration-150"
                style={{ color: active ? "#2563eb" : "var(--text-muted)", background: "none", border: "none", cursor: "pointer" }}
              >
                <span
                  className="flex items-center justify-center h-6 w-6 rounded-lg transition-colors duration-150"
                  style={{ background: active ? "#eff6ff" : "transparent" }}
                >
                  {item.icon}
                </span>
                <span className="text-[10px] font-semibold">{item.label}</span>
                {active && (
                  <span
                    className="absolute bottom-0 h-0.5 w-10 rounded-full"
                    style={{ background: "#2563eb" }}
                  />
                )}
              </button>
            );
          })}
        </nav>

      </div>
    </div>
  );
}
