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
    <div className="absolute -top-64 flex flex-col items-center z-[100] animate-pop-spin">
      <div className="bg-white rounded-[2.5rem] w-48 p-5 border-b-[10px] border-slate-200 shadow-2xl space-y-4 relative">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute -top-2 -right-2 bg-red-500 text-white p-1 rounded-lg border-b-2 border-red-700 active:translate-y-0.5"
        >
          <X size={14} />
        </button>

        <div className="text-center space-y-1">
          <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">
            Sign Lesson
          </p>
          <h3 className="font-display text-2xl font-black text-foreground">
            {sign.title}
          </h3>
        </div>

        {/* 3D Character Preview Placeholder Area */}
        <div className="aspect-square bg-slate-900 rounded-2xl flex items-center justify-center border-4 border-slate-100 shadow-inner overflow-hidden relative">
          <span className="text-5xl font-black text-white drop-shadow-lg">
            {sign.nepali_char}
          </span>
          {isCompleted && (
            <div className="absolute top-1 right-1 bg-yellow-400 p-1 rounded-full border-2 border-white">
              <Star size={12} className="fill-white text-white" />
            </div>
          )}
        </div>

        <GameButton
          variant={isCompleted ? "primary" : "retro"}
          size="sm"
          className="w-full py-1"
          onClick={() => (window.location.href = `/lessons/${sign.id}`)}
        >
          <div className="flex items-center gap-2">
            {isCompleted ? <RotateCcw size={16} /> : <BookOpen size={16} />}
            <span className="text-xs">{isCompleted ? "REVISE" : "LEARN"}</span>
          </div>
        </GameButton>
      </div>

      {/* Pointer triangle */}
      <div className="w-0 h-0 border-l-[12px] border-l-transparent border-r-[12px] border-r-transparent border-t-[18px] border-t-white drop-shadow-xl" />
    </div>
  );
}
