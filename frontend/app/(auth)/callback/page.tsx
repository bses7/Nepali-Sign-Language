"use client";

import { useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { tokenManager } from "@/lib/api/client";
import { Sparkles } from "lucide-react";

function CallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { setSocialLogin } = useAuthStore();

  useEffect(() => {
    const token = searchParams.get("token");
    const error = searchParams.get("error");

    if (token) {
      // Save token and sync Zustand
      tokenManager.setToken(token);
      setSocialLogin(token);

      // Short delay so the user can see the "beautiful" loading state
      const timer = setTimeout(() => {
        router.push("/dashboard");
      }, 1000);

      return () => clearTimeout(timer);
    } else if (error) {
      router.push(`/login?error=${error}`);
    }
  }, [searchParams, router, setSocialLogin]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background gap-6 relative overflow-hidden">
      {/* Background Decorative */}
      <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-primary/10 rounded-full blur-[120px] animate-pulse" />
      <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-secondary/10 rounded-full blur-[120px] animate-pulse" />

      <div className="relative">
        <div className="w-24 h-24 border-8 border-primary/20 border-t-primary rounded-full animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center">
          <Sparkles className="text-primary animate-pulse" size={32} />
        </div>
      </div>

      <div className="text-center space-y-2">
        <h2 className="font-display text-3xl font-black uppercase tracking-tighter text-primary">
          Syncing Profile
        </h2>
        <p className="text-muted-foreground text-xs font-black uppercase tracking-[0.2em] animate-pulse">
          Authenticating your account...
        </p>
      </div>
    </div>
  );
}

export default function AuthCallback() {
  return (
    <Suspense fallback={null}>
      <CallbackContent />
    </Suspense>
  );
}
