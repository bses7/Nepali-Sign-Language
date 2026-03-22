"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface GameButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "duolingo" | "glossy" | "retro" | "jelly" | "back";
  size?: "sm" | "md" | "lg";
  isLoading?: boolean;
  children: React.ReactNode;
}

export const GameButton = React.forwardRef<HTMLButtonElement, GameButtonProps>(
  (
    {
      className,
      variant = "primary",
      size = "md",
      isLoading = false,
      disabled,
      children,
      ...props
    },
    ref,
  ) => {
    const sizeClasses = {
      sm: "px-4 py-1 text-sm",
      md: variant === "back" ? "w-14 h-14 text-xl" : "px-8 py-3 text-xl",
      lg: variant === "back" ? "w-20 h-20 text-3xl" : "px-12 py-4 text-3xl",
    };

    const variantClasses = {
      primary: "bg-blue-500 text-white rounded-2xl active:scale-95",

      duolingo: cn(
        "bg-[#ff9600] text-white border-b-4 border-[#d97900] rounded-2xl transition-all duration-75",
        "active:translate-y-1 active:border-b-0",
      ),

      glossy: cn(
        "bg-[#ff8c00] text-white rounded-full transition-all duration-75",
        "shadow-[inset_0_4px_0_rgba(255,255,255,0.5),0_4px_0_#b35900]",
        "active:translate-y-1 active:shadow-[inset_0_2px_0_rgba(255,255,255,0.5),0_1px_0_#b35900]",
      ),

      // UPDATED JELLY STYLE (Matches your image exactly)
      jelly: cn(
        "relative p-0 rounded-full transition-all duration-75",
        "bg-[#4e901a] border-none shadow-[0_6px_0_0_#3d7014]", // Dark green bottom shadow
        "active:translate-y-[4px] active:shadow-[0_2px_0_0_#3d7014]",
      ),

      back: cn(
        "relative p-0 rounded-full transition-all duration-75",
        "bg-[#BFC9B4] border-b-4 border-[#2C3E33]/30 shadow-lg",
        "active:scale-95 active:translate-y-1 active:border-b-0",
      ),

      retro: cn(
        "relative group p-[6px] transition-all duration-75",
        "bg-[#235c24] border-2 border-[#0e2a0e] rounded-[20px]",
        "shadow-[0_6px_0_0_#133314]",
        "active:translate-y-[4px] active:shadow-[0_2px_0_0_#133314]",
      ),
    };

    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={cn(
          "inline-flex items-center justify-center select-none cursor-pointer font-black uppercase tracking-tighter",
          variantClasses[variant],
          className,
        )}
        {...props}
      >
        {variant === "retro" ? (
          <span
            className={cn(
              "w-full h-full flex items-center justify-center bg-[#b5e61d] border-2 border-[#0e2a0e] rounded-[14px] shadow-[inset_0_-4px_0_0_#89af16] text-[#0e2a0e]",
              sizeClasses[size],
            )}
          >
            {isLoading ? "..." : children}
          </span>
        ) : variant === "back" ? (
          <span
            className={cn(
              "flex items-center justify-center bg-[#5F7A61] rounded-full shadow-[inset_0_-4px_0_rgba(0,0,0,0.3),inset_0_4px_0_rgba(255,255,255,0.3)] overflow-hidden m-1.5 w-11 h-11 relative",
            )}
          >
            <div className="absolute top-1 left-2 w-3/4 h-1/2 bg-gradient-to-b from-white/40 to-transparent rounded-full rotate-[-15deg] blur-[0.5px]" />
            <div className="relative z-10 text-[#F4EDE4] drop-shadow-md">
              {isLoading ? "..." : children}
            </div>
          </span>
        ) : variant === "jelly" ? (
          <span
            className={cn(
              "w-full h-full flex items-center justify-center bg-[#76c92e] rounded-full relative overflow-hidden",
              "shadow-[inset_0_-4px_0_0_rgba(0,0,0,0.1),inset_0_4px_0_0_rgba(255,255,255,0.4)]",
              sizeClasses[size],
            )}
          >
            <div className="absolute top-1.5 left-4 w-6 h-3 bg-white/30 rounded-full blur-[1px] rotate-[-5deg]" />
            <div className="absolute top-3 left-10 w-2 h-2 bg-white/20 rounded-full blur-[0.5px]" />

            <span className="relative z-10 text-white drop-shadow-[0_2px_2px_rgba(0,0,0,0.3)]">
              {isLoading ? "..." : children}
            </span>
          </span>
        ) : (
          <span
            className={cn(
              sizeClasses[size],
              "flex items-center justify-center gap-3",
            )}
          >
            {isLoading ? "..." : children}
          </span>
        )}
      </button>
    );
  },
);

GameButton.displayName = "GameButton";
