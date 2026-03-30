"use client";

import { GameButton } from "./game-button";
import { Trophy, Zap, Coins, X, BrainCircuit } from "lucide-react";

export function QuizPreviewCard({ difficulty, category, onClose }: any) {
  return (
    <div className="relative flex flex-col items-center z-[200] animate-pop-spin">
      {/* Small version of the card */}
      <div className="bg-white rounded-[1.5rem] w-42 p-2.5 border-b-[6px] border-slate-200 shadow-2xl space-y-2 relative">
        <button
          onClick={onClose}
          className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-lg border-b-2 border-red-700 active:translate-y-0.5"
        >
          <X size={12} />
        </button>
        <div className="text-center space-y-1">
          <div className="inline-block bg-blue-100 p-3 rounded-2xl mb-2">
            <BrainCircuit className="text-blue-600" size={28} />
          </div>
          <h3 className="font-display text-md font-black text-foreground uppercase leading-none">
            {difficulty} Mastery
          </h3>
          <p className="text-[9px] font-bold text-muted-foreground uppercase tracking-widest">
            Challenge
          </p>
        </div>

        <GameButton
          variant="duolingo"
          size="sm"
          className="w-full py-1"
          onClick={() =>
            (window.location.href = `/quiz?category=${category}&difficulty=${difficulty}`)
          }
        >
          START QUIZ
        </GameButton>
      </div>
      <div className="w-0 h-0 border-l-[10px] border-l-transparent border-r-[10px] border-r-transparent border-t-[15px] border-t-white" />
    </div>
  );
}
