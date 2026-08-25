import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import { api } from "@/api/client";
import type { Provider } from "@/api/types";
import { Logo } from "@/components/logo";
import { ModeToggle } from "@/components/mode-toggle";
import { GitHubMark, GoogleMark } from "@/components/provider-marks";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/useAuth";

const MARKS: Record<string, typeof GoogleMark> = {
  google: GoogleMark,
  github: GitHubMark,
};

/**
 * Follows the shadcn login-block pattern: a centred card, providers above a
 * labelled divider, the email form below, and the mode switch as a plain link
 * rather than tabs.
 */
export function AuthPage() {
  const { login, register, authError, clearAuthError } = useAuth();
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [providers, setProviders] = useState<Provider[]>([]);

  useEffect(() => {
    let cancelled = false;
    api
      .providers()
      .then((next) => !cancelled && setProviders(next))
      .catch(() => undefined); // no providers configured is a normal state
    return () => {
      cancelled = true;
    };
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    clearAuthError();
    setBusy(true);
    try {
      if (isRegister) await register(email.trim(), password);
      else await login(email.trim(), password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  const shown = error ?? authError;

  return (
    <div className="bg-muted/40 flex min-h-full flex-col items-center justify-center gap-6 p-6 md:p-10">
      <div className="absolute top-4 right-4">
        <ModeToggle />
      </div>

      <div className="flex w-full max-w-sm flex-col gap-6">
        <div className="flex items-center justify-center gap-2 self-center font-medium">
          <div className="bg-primary text-primary-foreground flex size-7 items-center justify-center rounded-md">
            <Logo className="size-4" />
          </div>
          Sourcery
        </div>

        <Card>
          <CardHeader className="text-center">
            <CardTitle className="text-xl">
              {isRegister ? "Create your account" : "Welcome back"}
            </CardTitle>
            <CardDescription>
              {isRegister
                ? "Upload documents and ask questions with cited answers."
                : "Sign in to your documents and chats."}
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={submit} className="grid gap-6">
              {providers.length > 0 ? (
                <>
                  <div className="grid gap-3">
                    {providers.map((provider) => {
                      const Mark = MARKS[provider.id];
                      return (
                        <Button
                          key={provider.id}
                          type="button"
                          variant="outline"
                          size="lg"
                          className="w-full"
                          onClick={() => {
                            window.location.href = api.oauthStartUrl(provider.id);
                          }}
                        >
                          {Mark ? <Mark className="size-4" /> : null}
                          Continue with {provider.label}
                        </Button>
                      );
                    })}
                  </div>
                  <div className="after:border-border relative text-center text-sm after:absolute after:inset-0 after:top-1/2 after:z-0 after:flex after:items-center after:border-t">
                    <span className="bg-card text-muted-foreground relative z-10 px-2">
                      Or continue with
                    </span>
                  </div>
                </>
              ) : null}

              {shown ? (
                <Alert variant="destructive">
                  <AlertDescription>{shown}</AlertDescription>
                </Alert>
              ) : null}

              <div className="grid gap-3">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                />
              </div>

              <div className="grid gap-3">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  autoComplete={isRegister ? "new-password" : "current-password"}
                  required
                  minLength={8}
                  maxLength={72}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={isRegister ? "At least 8 characters" : undefined}
                />
              </div>

              <Button type="submit" size="lg" className="w-full" disabled={busy}>
                {busy ? <Loader2 className="animate-spin" /> : null}
                {isRegister ? "Create account" : "Sign in"}
              </Button>

              <div className="text-center text-sm">
                {isRegister ? "Already have an account?" : "Don't have an account?"}{" "}
                <button
                  type="button"
                  onClick={() => {
                    setIsRegister((v) => !v);
                    setError(null);
                    clearAuthError();
                  }}
                  className="underline underline-offset-4"
                >
                  {isRegister ? "Sign in" : "Sign up"}
                </button>
              </div>
            </form>
          </CardContent>
        </Card>

        <p className="text-muted-foreground text-center text-xs">
          Answers are drawn only from documents you upload.
        </p>
      </div>
    </div>
  );
}
