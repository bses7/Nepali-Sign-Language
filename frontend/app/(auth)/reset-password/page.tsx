"use client";

import { useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { GameButton } from "@/components/game-button";
import { GameHeader } from "@/components/game-header";
import { authService } from "@/lib/api/auth";
import { Lock, Sparkles, CheckCircle, ShieldCheck } from "lucide-react";
import { toast } from "sonner";

function ResetPasswordContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get("token");

  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!token) {
      toast.error("Invalid or expired reset link.");
      return;
    }

    if (password !== confirmPassword) {
      toast.error("Passwords do not match!");
      return;
    }

    setIsLoading(true);
    const response = await authService.resetPassword(password, token);

    if (response.success) {
      setIsSuccess(true);
      toast.success("Password updated successfully!");
      setTimeout(() => router.push("/login"), 3000);
    } else {
      toast.error(response.error || "Failed to reset password");
    }
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background px-4 py-20 relative overflow-hidden">
      {/* Background Decor */}
      <div className="absolute inset-0 pointer-events-none">
        <Sparkles
          className="absolute top-[10%] left-[20%] text-primary/20 animate-float"
          size={40}
        />
        <ShieldCheck
          className="absolute bottom-[20%] right-[15%] text-secondary/20 animate-wiggle"
          size={50}
        />
      </div>

      <div className="relative w-full max-w-md">
        <GameHeader
          title="SignLearn"
          subtitle="Secure your account"
          variant="duolingo"
        />

        <div className="glass rounded-[3rem] p-8 md:p-10 border-b-[12px] border-border/40 shadow-2xl mt-8 space-y-8">
          {!isSuccess ? (
            <>
              <div className="space-y-2 text-center">
                <h2 className="font-display text-3xl font-black uppercase tracking-tighter text-foreground">
                  New Password
                </h2>
                <p className="text-muted-foreground text-[10px] font-black uppercase tracking-[0.2em]">
                  Choose a strong secret
                </p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-xs font-black uppercase tracking-widest text-muted-foreground ml-2">
                      New Password
                    </label>
                    <div className="relative">
                      <Lock
                        className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/50"
                        size={18}
                      />
                      <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="••••••••"
                        className="game-input pl-12"
                        required
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-black uppercase tracking-widest text-muted-foreground ml-2">
                      Confirm Password
                    </label>
                    <div className="relative">
                      <Lock
                        className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/50"
                        size={18}
                      />
                      <input
                        type="password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="••••••••"
                        className="game-input pl-12"
                        required
                      />
                    </div>
                  </div>
                </div>

                <GameButton
                  type="submit"
                  variant="duolingo"
                  isLoading={isLoading}
                  className="w-full py-6 text-xl shadow-[0_8px_0_0_#46a302]"
                >
                  Update Password
                </GameButton>
              </form>
            </>
          ) : (
            <div className="text-center space-y-6 py-6 animate-in zoom-in">
              <div className="flex justify-center">
                <div className="bg-primary/10 p-6 rounded-full border-b-4 border-primary/20">
                  <CheckCircle size={64} className="text-primary" />
                </div>
              </div>
              <div className="space-y-2">
                <h3 className="text-2xl font-black uppercase tracking-tighter text-foreground">
                  Success!
                </h3>
                <p className="text-muted-foreground text-sm font-bold">
                  Your password has been updated. Returning you to the login
                  screen...
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Wrapper for Suspense (Required for useSearchParams in Next.js)
export default function ResetPasswordPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-background" />}>
      <ResetPasswordContent />
    </Suspense>
  );
}
