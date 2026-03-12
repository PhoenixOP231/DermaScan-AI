"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Message {
  role: "user" | "assistant";
  content: string;
}

// ─── Quick-action suggestions shown when chat is empty ────────────────────────

const SUGGESTIONS = [
  "How do I upload an image?",
  "What do the risk levels mean?",
  "How does the heatmap work?",
  "Which lesion classes can it detect?",
];

// ─── Tiny markdown-lite renderer (bold + code only) ──────────────────────────

function renderContent(text: string) {
  // Split on **…** for bold, `…` for inline code
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code
          key={i}
          className="px-1 py-0.5 rounded text-[11px] font-mono"
          style={{ background: "var(--bg-canvas)", border: "1px solid var(--border)" }}
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

// ─── Typing indicator (three animated dots) ───────────────────────────────────

function TypingDots() {
  return (
    <div className="flex items-center gap-1 px-1 py-0.5">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 rounded-full"
          style={{
            background: "var(--text-muted)",
            animation: `chat-dot 1.2s ease-in-out ${i * 0.2}s infinite`,
          }}
        />
      ))}
    </div>
  );
}

// ─── Main widget ──────────────────────────────────────────────────────────────

export default function ChatWidget() {
  const [open,     setOpen]     = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  const bottomRef  = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLTextAreaElement>(null);
  const windowRef  = useRef<HTMLDivElement>(null);

  // Scroll to latest message whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Focus input when panel opens
  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 120);
    }
  }, [open]);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;

      const userMsg: Message = { role: "user", content: trimmed };
      const next = [...messages, userMsg];
      setMessages(next);
      setInput("");
      setLoading(true);
      setError(null);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: trimmed,
            history: messages, // send history for multi-turn context
          }),
        });

        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(
            (detail as { error?: string }).error ?? `Server error ${res.status}`
          );
        }

        const data = await res.json() as { reply: string };
        setMessages([...next, { role: "assistant", content: data.reply }]);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Something went wrong. Please try again."
        );
        // Remove the user message on hard failure so they can retry
        setMessages(messages);
      } finally {
        setLoading(false);
      }
    },
    [messages, loading]
  );

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Keyframes injected once via a style tag */}
      <style>{`
        @keyframes chat-dot {
          0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
          40%            { transform: scale(1);   opacity: 1;   }
        }
        @keyframes chat-slide-up {
          from { opacity: 0; transform: translateY(16px) scale(0.97); }
          to   { opacity: 1; transform: translateY(0)    scale(1);    }
        }
        .chat-window { animation: chat-slide-up 0.22s cubic-bezier(0.25,1,0.5,1) forwards; }
        @keyframes chat-badge-pop {
          0%   { transform: scale(0); }
          70%  { transform: scale(1.2); }
          100% { transform: scale(1); }
        }
        .chat-badge { animation: chat-badge-pop 0.3s cubic-bezier(0.34,1.56,0.64,1) forwards; }
      `}</style>

      {/* ── Floating bubble ───────────────────────────────────────────────── */}
      <button
        aria-label={open ? "Close assistant" : "Open DermaScan assistant"}
        onClick={() => setOpen((v) => !v)}
        className="fixed bottom-5 right-5 z-50 h-14 w-14 rounded-full flex items-center justify-center shadow-lg transition-transform duration-150 active:scale-95 hover:scale-105"
        style={{
          background: open ? "var(--text-primary)" : "#2563eb",
          boxShadow : "0 4px 20px rgba(37,99,235,0.40)",
        }}
      >
        {/* Unread dot — shown before user first opens chat */}
        {!open && messages.length === 0 && (
          <span
            className="chat-badge absolute top-0 right-0 h-3.5 w-3.5 rounded-full bg-emerald-400"
            style={{ border: "2px solid #ffffff" }}
          />
        )}

        {open ? (
          /* Close X */
          <svg className="h-5 w-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        ) : (
          /* Chat bubble icon */
          <svg className="h-6 w-6 text-white" viewBox="0 0 24 24" fill="currentColor">
            <path d="M4.913 2.658c2.075-.27 4.19-.408 6.337-.408 2.147 0 4.262.139 6.337.408 1.922.25 3.291 1.861 3.405 3.727a4.403 4.403 0 0 0-1.032-.211 50.89 50.89 0 0 0-8.42 0c-2.358.196-4.04 2.19-4.04 4.434v4.286a4.47 4.47 0 0 0 2.433 3.984L7.28 21.53A.75.75 0 0 1 6 21v-4.03a48.527 48.527 0 0 1-1.087-.128C2.905 16.58 1.5 14.833 1.5 12.862V6.638c0-1.97 1.405-3.718 3.413-3.979Z" />
            <path d="M15.75 7.5c-1.376 0-2.739.057-4.086.169C10.124 7.797 9 9.103 9 10.609v4.285c0 1.507 1.128 2.814 2.67 2.94 1.243.102 2.5.157 3.768.165l2.782 2.781a.75.75 0 0 0 1.28-.53v-2.39l.33-.026c1.542-.125 2.67-1.433 2.67-2.94v-4.286c0-1.505-1.125-2.811-2.664-2.94A49.392 49.392 0 0 0 15.75 7.5Z" />
          </svg>
        )}
      </button>

      {/* ── Chat window ───────────────────────────────────────────────────── */}
      {open && (
        <div
          ref={windowRef}
          className="chat-window fixed bottom-24 right-5 z-50 flex flex-col"
          style={{
            width    : "min(380px, calc(100vw - 2.5rem))",
            height   : "min(540px, calc(100dvh - 8rem))",
            background: "var(--bg-surface)",
            border   : "1px solid var(--border)",
            borderRadius: "16px",
            boxShadow: "0 8px 40px rgba(15,25,35,0.18)",
            overflow : "hidden",
          }}
        >
          {/* Header */}
          <div
            className="shrink-0 flex items-center gap-3 px-4 py-3"
            style={{ background: "#2563eb" }}
          >
            <div
              className="h-8 w-8 rounded-full flex items-center justify-center shrink-0"
              style={{ background: "rgba(255,255,255,0.2)" }}
            >
              <svg className="h-4 w-4 text-white" viewBox="0 0 24 24" fill="currentColor">
                <path d="M16.5 7.5h-9v9h9v-9Z" />
                <path fillRule="evenodd" d="M8.25 2.25A.75.75 0 0 1 9 3v.75h2.25V3a.75.75 0 0 1 1.5 0v.75H15V3a.75.75 0 0 1 1.5 0v.75h.75a3 3 0 0 1 3 3v.75H21A.75.75 0 0 1 21 9h-.75v2.25H21a.75.75 0 0 1 0 1.5h-.75V15H21a.75.75 0 0 1 0 1.5h-.75v.75a3 3 0 0 1-3 3h-.75V21a.75.75 0 0 1-1.5 0v-.75h-2.25V21a.75.75 0 0 1-1.5 0v-.75H9V21a.75.75 0 0 1-1.5 0v-.75h-.75a3 3 0 0 1-3-3v-.75H3A.75.75 0 0 1 3 15h.75v-2.25H3a.75.75 0 0 1 0-1.5h.75V9H3a.75.75 0 0 1 0-1.5h.75v-.75a3 3 0 0 1 3-3h.75V3a.75.75 0 0 1 .75-.75ZM6 6.75A.75.75 0 0 1 6.75 6h10.5a.75.75 0 0 1 .75.75v10.5a.75.75 0 0 1-.75.75H6.75a.75.75 0 0 1-.75-.75V6.75Z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-semibold text-white leading-none">DermaScan Assistant</p>
              <p className="text-[11px] mt-0.5" style={{ color: "rgba(255,255,255,0.75)" }}>
                AI-powered user guide
              </p>
            </div>
            <div className="ml-auto flex items-center gap-1.5">
              <span
                className="h-2 w-2 rounded-full bg-emerald-400"
                style={{ boxShadow: "0 0 0 2px rgba(255,255,255,0.3)" }}
              />
              <span className="text-[11px]" style={{ color: "rgba(255,255,255,0.75)" }}>Online</span>
            </div>
          </div>

          {/* Message list */}
          <div
            className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3"
            style={{ background: "var(--bg-canvas)" }}
          >
            {/* Welcome message */}
            {messages.length === 0 && !loading && (
              <div className="flex flex-col gap-3">
                <div
                  className="self-start max-w-[88%] px-3.5 py-2.5 rounded-2xl rounded-tl-sm text-sm leading-relaxed"
                  style={{
                    background: "var(--bg-surface)",
                    border    : "1px solid var(--border)",
                    color     : "var(--text-primary)",
                  }}
                >
                  👋 Hi! I&apos;m the DermaScan AI guide. I can help you upload images,
                  understand diagnostic results, interpret risk levels, and explain how
                  the Grad-CAM heatmaps work. What would you like to know?
                </div>

                {/* Quick-action chips */}
                <div className="flex flex-col gap-1.5 mt-1">
                  <p className="text-[10px] font-semibold uppercase tracking-wide px-1" style={{ color: "var(--text-muted)" }}>
                    Suggested questions
                  </p>
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => send(s)}
                      className="self-start text-left text-xs px-3 py-1.5 rounded-full transition-colors duration-150"
                      style={{
                        background: "var(--bg-surface)",
                        border    : "1px solid var(--border)",
                        color     : "var(--blue)",
                      }}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLButtonElement).style.background = "#eff6ff";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLButtonElement).style.background = "var(--bg-surface)";
                      }}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Conversation */}
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className="max-w-[88%] px-3.5 py-2.5 text-sm leading-relaxed"
                  style={
                    msg.role === "user"
                      ? {
                          background  : "#2563eb",
                          color       : "#ffffff",
                          borderRadius: "16px 16px 4px 16px",
                        }
                      : {
                          background  : "var(--bg-surface)",
                          border      : "1px solid var(--border)",
                          color       : "var(--text-primary)",
                          borderRadius: "16px 16px 16px 4px",
                        }
                  }
                >
                  {msg.role === "assistant"
                    ? renderContent(msg.content)
                    : msg.content}
                </div>
              </div>
            ))}

            {/* Typing indicator */}
            {loading && (
              <div className="flex justify-start">
                <div
                  className="px-3.5 py-2.5"
                  style={{
                    background  : "var(--bg-surface)",
                    border      : "1px solid var(--border)",
                    borderRadius: "16px 16px 16px 4px",
                  }}
                >
                  <TypingDots />
                </div>
              </div>
            )}

            {/* Error */}
            {error && (
              <div
                className="text-xs px-3 py-2 rounded-xl"
                style={{ background: "var(--red-bg)", color: "var(--red)", border: "1px solid var(--red-bdr)" }}
              >
                {error}
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Input area */}
          <div
            className="shrink-0 px-3 py-3 flex items-end gap-2"
            style={{ borderTop: "1px solid var(--border)", background: "var(--bg-surface)" }}
          >
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                // Auto-expand up to ~4 rows
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 96)}px`;
              }}
              onKeyDown={onKeyDown}
              placeholder="Ask me anything…"
              rows={1}
              disabled={loading}
              className="flex-1 resize-none rounded-xl px-3 py-2.5 text-sm leading-snug outline-none transition-all duration-150"
              style={{
                background: "var(--bg-canvas)",
                border    : "1px solid var(--border)",
                color     : "var(--text-primary)",
                maxHeight : "96px",
                overflowY : "auto",
              }}
              onFocus={(e) => { e.target.style.borderColor = "#2563eb"; }}
              onBlur={(e)  => { e.target.style.borderColor = "var(--border)"; }}
            />
            <button
              onClick={() => send(input)}
              disabled={!input.trim() || loading}
              aria-label="Send message"
              className="shrink-0 h-9 w-9 rounded-xl flex items-center justify-center transition-all duration-150 active:scale-90"
              style={
                !input.trim() || loading
                  ? { background: "var(--bg-canvas)", cursor: "not-allowed" }
                  : { background: "#2563eb",           cursor: "pointer", boxShadow: "0 1px 4px rgba(37,99,235,0.35)" }
              }
            >
              <svg
                className="h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke={!input.trim() || loading ? "var(--text-muted)" : "#ffffff"}
                strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
              </svg>
            </button>
          </div>

          {/* Footer disclaimer */}
          <div
            className="shrink-0 px-4 py-1.5 text-center"
            style={{ borderTop: "1px solid var(--border)", background: "var(--bg-canvas)" }}
          >
            <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
              AI guide only · Not a substitute for medical advice
            </p>
          </div>
        </div>
      )}
    </>
  );
}
