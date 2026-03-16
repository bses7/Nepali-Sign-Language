"use client";

import { cn } from "@/lib/utils";
import { Lock, Sparkles } from "lucide-react";
import { useState, useEffect } from "react";

interface NodeProps {
  sign: any;
  index: number;
  isCurrent: boolean;
  onClick: (sign: any) => void;
}

export function RoadmapNode({ sign, index, isCurrent, onClick }: NodeProps) {
  const isLocked = sign.is_locked;
  const isCompleted = sign.is_completed;
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const getLevelLabel = (difficulty: string) => {
    switch (difficulty?.toLowerCase()) {
      case "easy":
        return "Lvl 1";
      case "medium":
        return "Lvl 2";
      case "hard":
        return "Lvl 3";
      default:
        return "Lvl 1";
    }
  };

  const colors = {
    // 1. Completed state (ORANGE)
    completed: "bg-[#ff9600] border-[#d97900] shadow-[0_6px_0_0_#b35900]",

    // 2. Current state (SAGE GREEN / PRIMARY)
    current:
      "bg-primary border-[#4a5f4b] shadow-[0_8px_0_0_#2C3E33] animate-pulse",

    // 3. Locked state (DARK SAGE)
    locked:
      "bg-[#2C3E33] border-[#1A261E] shadow-[0_4px_0_0_#111827] grayscale opacity-80",

    // 4. Unlocked but not current/completed (BLUE)
    active: "bg-[#76c92e] border-[#4e901a] shadow-[0_6px_0_0_#3d7014]",
  };

  // UPDATED LOGIC: Specific checks for each state
  const activeColor = isLocked
    ? colors.locked
    : isCurrent
      ? colors.current
      : isCompleted
        ? colors.completed
        : colors.active;

  if (!mounted) return null;

  return (
    <div className="relative flex flex-col items-center w-fit h-fit select-none">
      <div className="relative">
        <button
          disabled={isLocked}
          onClick={() => onClick(sign)}
          className={cn(
            "relative w-16 h-16 rounded-full transition-transform duration-75",
            "active:translate-y-1 active:shadow-none flex-shrink-0",
            activeColor,
            isCurrent && "hover:scale-105",
          )}
        >
          <div
            className={cn(
              "absolute inset-1.5 rounded-full flex items-center justify-center overflow-hidden border border-white/10",
              isLocked ? "bg-black/20" : "bg-black/5",
            )}
          >
            <span
              className={cn(
                "font-display font-black text-2xl",
                isLocked ? "text-slate-500" : "text-white drop-shadow-md",
              )}
            >
              {sign.nepali_char}
            </span>

            {!isLocked && (
              <>
                <div className="absolute top-1 left-3 w-6 h-3 bg-white/30 rounded-full blur-[0.5px] rotate-[-15deg]" />
                <div className="absolute bottom-1.5 right-3 w-2 h-2 bg-white/10 rounded-full blur-[0.5px]" />
              </>
            )}
          </div>
        </button>

        {isLocked && (
          <div className="absolute -top-1 -right-1 z-20 pointer-events-none">
            <div className="bg-slate-100 p-1 rounded-lg border-2 border-slate-300 shadow-md rotate-12">
              <Lock size={10} className="text-slate-500" />
            </div>
          </div>
        )}

        {isCurrent && (
          <Sparkles
            className="absolute -top-1 -right-1 text-yellow-400 animate-pulse z-20"
            size={20}
          />
        )}
      </div>

      <div
        className={cn(
          "mt-3 px-3 py-0.5 rounded-xl border-b-2 font-black text-[8px] uppercase tracking-tighter shadow-sm whitespace-nowrap",
          isLocked
            ? "bg-slate-100 text-slate-400 border-slate-200"
            : "bg-white text-primary border-slate-200",
        )}
      >
        {getLevelLabel(sign.difficulty)}
      </div>

      {isCurrent && (
        <div className="absolute top-0 w-16 h-16 -z-10 bg-primary/20 rounded-full animate-ping scale-110 opacity-20" />
      )}
    </div>
  );
}
