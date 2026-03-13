import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  convertToModelMessages,
  type UIMessage,
} from "ai";

const VITE_BACKEND_URL = process.env.VITE_BACKEND_URL ?? "http://localhost:8000/message";
console.log("Using backend URL:", VITE_BACKEND_URL);

function extractAssistantText(payload: unknown): string {
  if (typeof payload === "string") {
    return payload;
  }

  if (payload && typeof payload === "object") {
    const record = payload as Record<string, unknown>;
    const candidateKeys = [
      "text",
      "message",
      "response",
      "output_text",
      "content",
      "answer",
    ];

    for (const key of candidateKeys) {
      const value = record[key];
      if (typeof value === "string" && value.trim().length > 0) {
        return value;
      }
    }
  }

  return JSON.stringify(payload, null, 2);
}

export async function POST(req: Request) {
  const {
    messages,
    system,
    model
  }: {
    messages: UIMessage[];
    system?: string;
    model?: string;
  } = await req.json();

  try {
    const backendResponse = await fetch(VITE_BACKEND_URL + "/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: model,
        messages: await convertToModelMessages(messages),
        system,
      }),
    });

    if (!backendResponse.ok) {
      throw new Error(`Backend API error: ${backendResponse.status} ${backendResponse.statusText}`);
    }

    const backendJson = await backendResponse.json();
    const assistantText = extractAssistantText(backendJson);

    const stream = createUIMessageStream({
      execute: ({ writer }) => {
        const textId = crypto.randomUUID();
        writer.write({ type: "start" });
        writer.write({ type: "text-start", id: textId });
        writer.write({ type: "text-delta", id: textId, delta: assistantText });
        writer.write({ type: "text-end", id: textId });
        writer.write({ type: "finish", finishReason: "stop" });
      },
    });

    return createUIMessageStreamResponse({ stream });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown backend error";
    return new Response(message, { status: 500 });
  }
}
