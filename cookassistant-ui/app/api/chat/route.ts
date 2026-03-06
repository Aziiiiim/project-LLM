import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  convertToModelMessages,
  type UIMessage,
} from "ai";

const { loadEnvFile } = require('node:process');
loadEnvFile('../.env'); // Loads .env from parent directory

const MODEL_NAME = process.env.AI_MODEL ?? "gpt-5-nano";
console.log("Using model:", MODEL_NAME);
const API_ENDPOINT = process.env.AI_ENDPOINT;

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
  }: {
    messages: UIMessage[];
    system?: string;
  } = await req.json();

  try {
    const backendResponse = await fetch("http://localhost:8000/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL_NAME,
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
