import { NextRequest, NextResponse } from "next/server";

interface GeminiMessage {
  role: "user" | "model";
  parts: { text: string }[];
}

const SYSTEM_PROMPT = `You are DermaScan AI Assistant, a knowledgeable and friendly assistant for the DermaScan AI platform — an AI-powered skin lesion classifier.

The platform uses a ResNet18 model trained on the HAM10000 dataset to classify dermoscopy images into 7 categories:
- Melanocytic Nevi (nv) — Benign
- Melanoma (mel) — Malignant
- Benign Keratosis-like Lesions (bkl) — Benign
- Basal Cell Carcinoma (bcc) — Malignant
- Actinic Keratoses (akiec) — Pre-cancerous
- Vascular Lesions (vasc) — Benign
- Dermatofibroma (df) — Benign

The model uses Grad-CAM (Class Activation Mapping) for visual explanations, and runs on ONNX Runtime for fast serverless inference.

Help users understand:
- Their analysis results and what the predictions mean
- The confidence scores and what they indicate
- Each skin lesion type (causes, appearance, risk level)
- How Grad-CAM heatmaps work
- General dermoscopy and skin health concepts

Always end responses with a reminder that this tool is for educational purposes only and users should consult a qualified dermatologist for any medical concerns. Keep answers concise, friendly, and clear.`;

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json();

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: "Gemini API key is not configured on the server." },
        { status: 500 }
      );
    }

    const geminiMessages: GeminiMessage[] = messages.map(
      (m: { role: string; content: string }) => ({
        role   : m.role === "assistant" ? "model" : "user",
        parts  : [{ text: m.content }],
      })
    );

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`,
      {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({
          system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
          contents          : geminiMessages,
          generationConfig  : {
            maxOutputTokens: 512,
            temperature    : 0.7,
          },
        }),
      }
    );

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      const msg = (err as { error?: { message?: string } }).error?.message ?? `Gemini API error ${response.status}`;
      return NextResponse.json({ error: msg }, { status: 502 });
    }

    const data = await response.json();
    const reply: string =
      data?.candidates?.[0]?.content?.parts?.[0]?.text ??
      "Sorry, I couldn't generate a response. Please try again.";

    return NextResponse.json({ reply });
  } catch {
    return NextResponse.json(
      { error: "An unexpected error occurred. Please try again." },
      { status: 500 }
    );
  }
}
