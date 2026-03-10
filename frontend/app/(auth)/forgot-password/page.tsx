"use client";

import { useState } from "react";
import Link from "next/link";
import { GameButton } from "@/components/game-button";
import { GameHeader } from "@/components/game-header";
import { authService } from "@/lib/api/auth";
import { Mail, ArrowLeft, Sparkles, CheckCircle, Send } from "lucide-react";
import { toast } from "sonner";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    const response = await authService.recoverPassword(email);

    if (response.success) {
      setIsSubmitted(true);
      toast.success("Recovery instructions sent to your email!");
    } else {
      toast.error(response.error || "Failed to send recovery email");
    }
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background px-4 py-20 relative overflow-hidden">
      {/* BACKGROUND ELEMENTS */}
      <div className="absolute inset-0 pointer-events-none">
        <Sparkles
          className="absolute top-[15%] right-[10%] text-primary/20 animate-float"
          size={40}
        />
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-primary/5 rounded-full blur-[120px]" />
      </div>

      <div className="relative w-full max-w-md">
        <GameHeader
          title="SignLearn"
          subtitle="Account Recovery"
          variant="duolingo"
        />

        <div className="glass rounded-[3rem] p-8 md:p-10 border-b-[12px] border-border/40 shadow-2xl mt-8 space-y-8">
          {!isSubmitted ? (
            <>
              <div className="space-y-2 text-center">
                <h2 className="font-display text-3xl font-black uppercase tracking-tighter text-foreground">
                  Lost your key?
                </h2>
                <p className="text-muted-foreground text-[10px] font-black uppercase tracking-[0.2em]">
                  Enter your email to reset
                </p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
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
                    />
                  </div>
                </div>

                <GameButton
                  type="submit"
                  variant="duolingo"
                  isLoading={isLoading}
                  className="w-full mt-1"
                >
                  Send Reset Link
                </GameButton>
              </form>
            </>
          ) : (
            <div className="text-center space-y-6 py-4 animate-in fade-in zoom-in">
              <div className="flex justify-center">
                <div className="bg-primary/10 p-6 rounded-full border-b-4 border-primary/20">
                  <CheckCircle size={64} className="text-primary" />
                </div>
              </div>
              <div className="space-y-2">
                <h3 className="text-2xl font-black uppercase tracking-tighter text-foreground">
                  Email Sent!
                </h3>
                <p className="text-muted-foreground text-sm font-bold">
                  Check <span className="text-primary">{email}</span> for
                  instructions to get back in the game.
                </p>
              </div>
            </div>
          )}

          <div className="pt-2">
            <Link
              href="/login"
              className="flex items-center justify-center gap-2 text-xs font-black uppercase tracking-widest text-muted-foreground hover:text-primary transition-colors group"
            >
              <ArrowLeft
                size={14}
                className="group-hover:-translate-x-1 transition-transform"
              />
              Back to Login
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
