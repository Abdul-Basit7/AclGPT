import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { api } from "@/api/client";
import type { User } from "@/api/types";

const TOKEN_KEY = "sourcery.token";

interface AuthValue {
  token: string | null;
  user: User | null;
  ready: boolean;
  /** Set when an OAuth redirect came back with a failure. */
  authError: string | null;
  clearAuthError: () => void;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthValue | null>(null);

/**
 * Read a token or error handed back by the OAuth callback.
 *
 * The backend redirects to `/#token=…` or `/#error=…`. Using the fragment keeps
 * the token out of server logs, and reading it here avoids pulling in a router
 * for what is a single one-shot callback.
 */
function consumeAuthFragment(): { token?: string; error?: string } {
  const raw = window.location.hash.replace(/^#/, "");
  if (!raw) return {};

  const params = new URLSearchParams(raw);
  const token = params.get("token") ?? undefined;
  const error = params.get("error") ?? undefined;
  if (!token && !error) return {};

  params.delete("token");
  params.delete("error");
  const rest = params.toString();
  window.history.replaceState(
    null,
    "",
    `${window.location.pathname}${window.location.search}${rest ? `#${rest}` : ""}`,
  );
  return { token, error };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    // A token in the URL fragment is fresher than anything already stored.
    const fromRedirect = consumeAuthFragment();
    if (fromRedirect.error) setAuthError(fromRedirect.error);
    if (fromRedirect.token) {
      try {
        localStorage.setItem(TOKEN_KEY, fromRedirect.token);
      } catch {
        /* storage blocked: the session lasts until reload */
      }
    }

    let stored: string | null = fromRedirect.token ?? null;
    if (!stored) {
      try {
        stored = localStorage.getItem(TOKEN_KEY);
      } catch {
        stored = null;
      }
    }

    if (!stored) {
      setReady(true);
      return;
    }

    // Validate before trusting it, so an expired session logs out cleanly.
    api
      .me(stored)
      .then((me) => {
        if (cancelled) return;
        setUser(me);
        setToken(stored);
      })
      .catch(() => {
        if (cancelled) return;
        try {
          localStorage.removeItem(TOKEN_KEY);
        } catch {
          /* ignore */
        }
        setToken(null);
      })
      .finally(() => {
        if (!cancelled) setReady(true);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const accept = useCallback((accessToken: string, nextUser: User) => {
    try {
      localStorage.setItem(TOKEN_KEY, accessToken);
    } catch {
      /* ignore */
    }
    setToken(accessToken);
    setUser(nextUser);
    setAuthError(null);
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      const result = await api.login(email, password);
      accept(result.access_token, result.user);
    },
    [accept],
  );

  const register = useCallback(
    async (email: string, password: string) => {
      const result = await api.register(email, password);
      accept(result.access_token, result.user);
    },
    [accept],
  );

  const logout = useCallback(() => {
    try {
      localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* ignore */
    }
    setToken(null);
    setUser(null);
  }, []);

  const clearAuthError = useCallback(() => setAuthError(null), []);

  const value = useMemo<AuthValue>(
    () => ({
      token,
      user,
      ready,
      authError,
      clearAuthError,
      login,
      register,
      logout,
    }),
    [token, user, ready, authError, clearAuthError, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside an AuthProvider");
  return context;
}
