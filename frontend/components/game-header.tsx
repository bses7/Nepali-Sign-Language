"use client";

import React from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { Sparkles } from "lucide-react";

interface GameHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string;
  subtitle?: string;
  variant?: "primary" | "duolingo" | "retro";
}

export const GameHeader = React.forwardRef<HTMLDivElement, GameHeaderProps>(
  ({ title, subtitle, variant = "primary", className, ...props }, ref) => {
    const textVariants = {
      primary: "text-blue-500 drop-shadow-[0_6px_0_#1d4ed8]",
      duolingo: "text-[#58cc02] drop-shadow-[0_6px_0_#46a302]",
      retro: "text-[#b5e61d] drop-shadow-[4px_4px_0_#0e2a0e]",
    };

    return (
      <div
        ref={ref}
        className={cn("text-center mb-12 space-y-6", className)}
        {...props}
      >
        <Link href="/" className="inline-block group">
          <div className="relative">
            {/* 1. CONSTANTLY ANIMATED BADGE */}
            <div
              className={cn(
                "absolute -top-8 -right-10 p-2 rounded-2xl z-10",
                "bg-yellow-400 text-white shadow-[0_4px_0_0_#cc9900]",
                "animate-float", // Continuous floating
              )}
            >
              <Sparkles
                size={24}
                fill="currentColor"
                className="animate-sparkle-pulse" // Continuous spinning/scaling
              />
            </div>

            {/* 2. CONSTANTLY FLOATING TITLE */}
            <div className="animate-float" style={{ animationDuration: "4s" }}>
              <h1
                className={cn(
                  "font-display text-7xl font-black uppercase tracking-tighter transition-all duration-75",
                  "group-active:translate-y-2 group-active:drop-shadow-none",
                  textVariants[variant],
                )}
              >
                {title}
              </h1>
            </div>

            {/* 3. REFLECTION SHINE (Bonus: makes it look glossy) */}
            <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none opacity-30">
              <div className="absolute -inset-full w-[200%] h-[200%] bg-linear-to-tr from-transparent via-white/60 to-transparent animate-shine" />
            </div>
          </div>
        </Link>

        {/* Subtitle with a gentle wiggle */}
        {subtitle && (
          <div className="flex flex-col items-center animate-wiggle">
            <div
              className={cn(
                "px-8 py-2 rounded-2xl font-black text-sm uppercase tracking-widest shadow-md",
                variant === "duolingo"
                  ? "bg-[#afdf00] text-[#46a302] border-b-4 border-[#84a900]"
                  : "bg-white text-primary border-b-4 border-slate-200",
              )}
            >
              {subtitle}
            </div>
          </div>
        )}
      </div>
    );
  },
);

GameHeader.displayName = "GameHeader";
