"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { GameButton } from "@/components/game-button";
import { useAuthStore } from "@/lib/store/auth";
import { GameHeader } from "@/components/game-header";
import { FcGoogle } from "react-icons/fc";
import { FaGithub } from "react-icons/fa";
import { Mail, Lock, Sparkles, Fingerprint, GraduationCap } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const { login, isLoading, error, isAuthenticated, clearError } =
    useAuthStore();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [localError, setLocalError] = useState("");

  useEffect(() => {
    if (isAuthenticated) {
      router.push("/dashboard");
    }
  }, [isAuthenticated, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError("");
    clearError();

    if (!email || !password) {
      setLocalError("Please fill in all fields");
      return;
    }

    const success = await login(email, password);
    if (!success) {
      setLocalError(error || "Login failed. Please try again.");
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background px-4 py-20 relative overflow-hidden">
      {/* 1. FLOATING BACKGROUND ELEMENTS (Matching Signup) */}
      <div className="absolute inset-0 pointer-events-none">
        <Sparkles
          className="absolute top-[15%] left-[10%] text-primary/20 animate-float"
          size={40}
        />
        <Fingerprint
          className="absolute bottom-[20%] left-[15%] text-secondary/20 animate-wiggle"
          size={50}
        />
        <GraduationCap
          className="absolute top-[25%] right-[12%] text-accent/20 animate-float"
          size={45}
        />
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-secondary/5 rounded-full blur-[120px]" />
      </div>

      <div className="relative w-full max-w-md">
        {/* Header */}
        <GameHeader
          title="SignLearn"
          subtitle="Welcome back, traveler!"
          variant="duolingo"
        />

        {/* Form Card */}
        <div className="glass rounded-[3rem] p-8 md:p-10 border-b-[12px] border-border/40 shadow-2xl mt-8 space-y-8">
          <div className="space-y-2 text-center">
            <h2 className="font-display text-3xl font-black uppercase tracking-tighter text-foreground">
              Sign In
            </h2>
            <p className="text-muted-foreground text-[10px] font-black uppercase tracking-[0.2em]">
              Enter your credentials
            </p>
          </div>

          {/* Error Message */}
          {(localError || error) && (
            <div className="bg-destructive/10 border-b-4 border-destructive text-destructive px-6 py-3 rounded-2xl text-sm font-bold animate-wiggle text-center">
              {localError || error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email Input */}
            <div className="space-y-2">
              <label className="text-xs font-black uppercase tracking-widest text-muted-foreground ml-2">
                Email Address
              </label>
              <div className="relative">
                <Mail
                  className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/50"
                  size={18}
                />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="learner@example.com"
                  className="game-input pl-12"
                  required
                  disabled={isLoading}
                />
              </div>
            </div>

            {/* Password Input */}
            <div className="space-y-2">
              {/* Inside your Password Input div on the Login Page */}
              <div className="flex justify-between items-center ml-2">
                <label className="text-xs font-black uppercase tracking-widest text-muted-foreground">
                  Password
                </label>
                <Link
                  href="/forgot-password"
                  className="text-[10px] font-black uppercase text-primary hover:underline underline-offset-4"
                >
                  Forgot?
                </Link>
              </div>
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
                  disabled={isLoading}
                />
              </div>
            </div>

            <div className="pt-2">
              <GameButton
                type="submit"
                variant="retro"
                isLoading={isLoading}
                className="w-full mt-1"
              >
                Continue
              </GameButton>
            </div>
          </form>

          {/* Social Divider */}
          <div className="flex items-center gap-4 ">
            <div className="flex-1 h-1 bg-border/30 rounded-full" />
            <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">
              Or
            </span>
            <div className="flex-1 h-1 bg-border/30 rounded-full" />
          </div>

          {/* Social Login */}
          <div className="grid grid-cols-2 gap-4">
            <button
              type="button"
              onClick={() =>
                (window.location.href =
                  "http://localhost:8000/api/v1/auth/login/google")
              }
              className="flex items-center justify-center gap-2 bg-white border-b-4 border-gray-200 active:border-b-0 active:translate-y-1 py-3 rounded-2xl font-bold transition-all hover:bg-gray-50 text-sm"
            >
              <FcGoogle className="text-xl" />
              Google
            </button>

            <button
              type="button"
              onClick={() =>
                (window.location.href =
                  "http://localhost:8000/api/v1/auth/login/github")
              }
              className="flex items-center justify-center gap-2 bg-slate-900 text-white border-b-4 border-slate-700 active:border-b-0 active:translate-y-1 py-3 rounded-2xl font-bold transition-all hover:bg-slate-800 text-sm"
            >
              <FaGithub className="text-xl" />
              GitHub
            </button>
          </div>

          {/* Sign Up Link */}
          <p className="text-center text-sm font-bold text-muted-foreground">
            New here?{" "}
            <Link
              href="/signup"
              className="text-primary hover:underline underline-offset-4"
            >
              Create an account
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
