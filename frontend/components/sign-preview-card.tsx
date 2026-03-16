"use client";

import { GameButton } from "./game-button";
import { cn } from "@/lib/utils";
import { BookOpen, RotateCcw, X, Star } from "lucide-react";

export function SignPreviewCard({
  sign,
  onClose,
}: {
  sign: any;
  onClose: () => void;
}) {
  const isCompleted = sign.is_completed;

  return (
    <div className="relative flex flex-col items-center z-[100] animate-pop-spin">
      {/* Small version of the card */}
      <div className="bg-white rounded-[1.5rem] w-32 p-2.5 border-b-[6px] border-slate-200 shadow-2xl space-y-2 relative">
        <button
          onClick={onClose}
          className="absolute -top-1 -right-1 bg-red-500 text-white p-1 rounded-lg border-b-2 border-red-700 active:translate-y-0.5"
        >
          <X size={10} />
        </button>

        <div className="text-center">
          <h3 className="font-display text-sm font-black text-foreground leading-none uppercase">
            {sign.title}
          </h3>
        </div>

        <div className="aspect-square bg-slate-900 rounded-lg flex items-center justify-center border-2 border-slate-100 shadow-inner overflow-hidden relative">
          <span className="text-2xl font-black text-white">
            {sign.nepali_char}
          </span>
        </div>

        <GameButton
          variant={isCompleted ? "duolingo" : "retro"}
          size="sm"
          className="w-full py-1 "
          onClick={() => (window.location.href = `/lessons/${sign.id}`)}
        >
          <span className="text-[9px] font-black">
            {isCompleted ? "REVISE" : "LEARN"}
          </span>
        </GameButton>
      </div>

      {/* Pointer matches the card style */}
      <div className="w-0 h-0 border-l-[8px] border-l-transparent border-r-[8px] border-r-transparent border-t-[10px] border-t-white" />
    </div>
  );
}
