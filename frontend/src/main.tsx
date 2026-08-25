import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ThemeProvider } from "next-themes";

import App from "./App";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AuthProvider } from "@/hooks/useAuth";
import "./index.css";

const container = document.getElementById("root");
if (!container) throw new Error("Missing #root element");

createRoot(container).render(
  <StrictMode>
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      storageKey="sourcery.theme"
      disableTransitionOnChange
    >
      <TooltipProvider delayDuration={400}>
        <AuthProvider>
          <App />
          <Toaster position="bottom-right" />
        </AuthProvider>
      </TooltipProvider>
    </ThemeProvider>
  </StrictMode>,
);
