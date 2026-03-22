"use client";

import { Hammer, RefreshCcw, WifiOff } from "lucide-react";
import { GameButton } from "./game-button";

export function MaintenanceScreen() {
  return (
    <div className="fixed inset-0 z-[10000] bg-[#F4EDE4] flex flex-col items-center justify-center p-6 text-center">
      <div className="absolute w-[500px] h-[500px] bg-primary/10 rounded-full blur-[120px] animate-pulse" />

      <div className="relative space-y-8 max-w-md">
        <div className="relative w-40 h-40 mx-auto">
          <div className="absolute inset-0 border-[10px] border-dashed border-primary/30 rounded-full animate-[spin_10s_linear_infinite]" />

          <div className="absolute inset-0 flex items-center justify-center">
            <div className="bg-white p-6 rounded-full border-b-8 border-slate-200 shadow-2xl animate-bounce">
              <Hammer size={48} className="text-primary" />
            </div>
          </div>

          <div className="absolute -top-2 -right-2 bg-yellow-400 p-2 rounded-xl rotate-12 shadow-lg animate-pulse">
            <WifiOff size={20} className="text-white" />
          </div>
        </div>

        <div className="space-y-3">
          <h1 className="font-display text-4xl font-black uppercase tracking-tighter text-[#2C3E33]">
            Under <span className="text-primary">Maintenance</span>
          </h1>
          <p className="text-muted-foreground font-bold leading-relaxed">
            Our servers are currently offline for a scheduled power-up.
            We'll be back in the game shortly!
          </p>
        </div>

        <div className="pt-4">
          <GameButton
            variant="duolingo"
            className="w-full py-6 text-xl"
            onClick={() => window.location.reload()}
          >
            <RefreshCcw className="mr-2 h-5 w-5" />
            RETRY CONNECTION
          </GameButton>
        </div>

        <p className="text-[10px] font-black uppercase tracking-[0.3em] text-muted-foreground/50">
          Error Code: SERVER_UNREACHABLE_01
        </p>
      </div>
    </div>
  );
}
