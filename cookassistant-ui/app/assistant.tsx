"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import {
  useChatRuntime,
  AssistantChatTransport,
} from "@assistant-ui/react-ai-sdk";
import { lastAssistantMessageIsCompleteWithToolCalls } from "ai";
import { Thread } from "@/components/assistant-ui/thread";
import { useState } from "react";

export const Assistant = () => {
  const [modelName, setModelName] = useState("gpt-4.1-mini");
  const runtime = useChatRuntime({
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
    transport: new AssistantChatTransport({
      api: "/api/chat",
      body: { model: modelName }
    }),
  });

  return (
    <>
      <div className="fixed left-4 top-4 z-30">
        <select
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          className="min-w-44 cursor-pointer rounded-md border border-input bg-card px-3 py-2 pr-8 text-sm text-card-foreground shadow-sm transition-colors hover:border-ring focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <option value="gpt-4.1-mini" className="bg-popover text-popover-foreground">
            gpt-4.1-mini
          </option>
          <option value="gpt-4.1-nano" className="bg-popover text-popover-foreground">
            gpt-4.1-nano
          </option>
        </select>
      </div>
      <AssistantRuntimeProvider runtime={runtime}>
        <div className="h-dvh">
          <Thread />
        </div>
      </AssistantRuntimeProvider>
    </>
  );
};
