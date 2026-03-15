"use client";

import { useState, useEffect } from "react";
import { useAuthStore } from "@/lib/store/auth";
import { lessonsService } from "@/lib/api/lessons";
import { GameButton } from "@/components/game-button";
import { GameHeader } from "@/components/game-header";
import { Roadmap3D } from "@/components/roadmap";
import { CoinDisplay } from "@/components/game-stats";
import { GameShopIcon } from "@/components/icons/game-shop-icon";
import {
  GraduationCap,
  BookOpen,
  ArrowLeft,
  HelpCircle,
  X,
  Lightbulb,
} from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";

export default function LessonsPage() {
  const [selectedCategory, setSelectedCategory] = useState<
    "vowel" | "consonant" | null
  >(null);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [signs, setSigns] = useState<any[]>([]);
  const { isAuthenticated, dashboard, fetchDashboard } = useAuthStore();
  const [selectedSign, setSelectedSign] = useState<any>(null);

  useEffect(() => {
    if (isAuthenticated) {
      fetchDashboard();
      lessonsService
        .getSigns()
        .then((res) => res.success && setSigns(res.data || []));
    }
  }, [isAuthenticated, fetchDashboard]);

  const filteredSigns = signs.filter((s) => s.category === selectedCategory);

  if (!selectedCategory) {
    return (
      <div className="min-h-screen bg-[#F4EDE4] flex flex-col items-center justify-center p-6">
        <GameHeader
          title="SignLearn"
          subtitle="Pick your training ground"
          variant="duolingo"
        />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl mt-10">
          <div
            onClick={() => setSelectedCategory("vowel")}
            className="group cursor-pointer bg-white rounded-[3rem] p-10 border-b-[12px] border-slate-200 hover:border-primary transition-all hover:-translate-y-2 flex flex-col items-center gap-6"
          >
            <div className="bg-primary/20 p-8 rounded-full border-b-8 border-primary/30 group-hover:scale-110 transition-transform">
              <GraduationCap size={80} className="text-primary" />
            </div>
            <div className="text-center">
              <h2 className="text-4xl font-black uppercase tracking-tighter text-foreground">
                Vowels
              </h2>
              <p className="text-muted-foreground font-bold uppercase text-xs tracking-widest mt-2">
                13 Signs to Master
              </p>
            </div>
            <GameButton
              variant="duolingo"
              className="w-full pointer-events-none"
            >
              Enter World
            </GameButton>
          </div>
          <div
            onClick={() => setSelectedCategory("consonant")}
            className="group cursor-pointer bg-white rounded-[3rem] p-10 border-b-[12px] border-slate-200 hover:border-secondary transition-all hover:-translate-y-2 flex flex-col items-center gap-6"
          >
            <div className="bg-secondary/20 p-8 rounded-full border-b-8 border-secondary/30 group-hover:scale-110 transition-transform">
              <BookOpen size={80} className="text-secondary" />
            </div>
            <div className="text-center">
              <h2 className="text-4xl font-black uppercase tracking-tighter text-foreground">
                Consonants
              </h2>
              <p className="text-muted-foreground font-bold uppercase text-xs tracking-widest mt-2">
                36 Signs to Master
              </p>
            </div>
            <GameButton variant="jelly" className="w-full pointer-events-none">
              Enter World
            </GameButton>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full relative bg-[#7eb34d] overflow-hidden">
      {/* HUD: TOP NAV */}
      <div className="absolute top-6 left-24 z-50 flex items-center gap-6">
        <GameButton
          variant="back"
          onClick={() => setSelectedCategory(null)}
          className="shadow-2xl"
        >
          <ArrowLeft size={24} strokeWidth={3} />
        </GameButton>

        <div className="flex items-center gap-4">
          <CoinDisplay amount={dashboard?.coins ?? 0} />

          {/* RESTORED SHOP BUTTON STYLE */}
          <Link href="/shop" className="group">
            <button
              className={cn(
                "relative flex items-center gap-3 px-5 py-2 rounded-2xl transition-all duration-75",
                "bg-white border-b-4 border-slate-200 hover:border-blue-400 active:border-b-0 active:translate-y-1 shadow-xl",
              )}
            >
              <div className="w-8 h-8 group-hover:scale-110 group-hover:rotate-6 transition-transform">
                <GameShopIcon />
              </div>
              <div className="flex flex-col items-start leading-none hidden sm:flex">
                <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                  Market
                </span>
                <span className="text-sm font-black uppercase text-blue-600">
                  Shop
                </span>
              </div>
            </button>
          </Link>
        </div>
      </div>

      {/* THREE.JS ROADMAP ENGINE */}
      <Roadmap3D
        signs={filteredSigns}
        avatarFolder={dashboard?.equipped_avatar_folder}
        selectedSign={selectedSign} // Pass selected sign
        onLevelClick={(sign: any) => setSelectedSign(sign)} // Set selected on click
      />

      {/* HELP BUTTON */}
      <div className="absolute bottom-10 right-10 z-50">
        <button
          onClick={() => setIsHelpOpen(true)}
          className="w-16 h-16 bg-yellow-400 border-b-8 border-yellow-600 rounded-full flex items-center justify-center text-white shadow-2xl hover:scale-110 active:translate-y-2 active:border-b-0 transition-all group"
        >
          <HelpCircle
            size={32}
            className="group-hover:rotate-12 transition-transform"
          />
        </button>
      </div>

      {isHelpOpen && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center p-6 bg-black/60 backdrop-blur-md">
          <div className="bg-white rounded-[3rem] w-full max-w-md p-10 border-b-[12px] border-slate-200 shadow-2xl relative animate-pop-spin">
            {/* Close Button */}
            <button
              onClick={() => setIsHelpOpen(false)}
              className="absolute -top-4 -right-4 bg-red-500 text-white p-2 rounded-xl border-b-4 border-red-700 active:translate-y-1 active:border-b-0 transition-all z-[10000]"
            >
              <X size={24} />
            </button>

            <div className="text-center space-y-6">
              <div className="inline-block bg-yellow-100 p-5 rounded-[2rem] border-b-4 border-yellow-200">
                <Lightbulb size={48} className="text-yellow-500" />
              </div>

              <h3 className="font-display text-4xl font-black uppercase tracking-tighter text-foreground leading-none">
                Map Guide
              </h3>

              <div className="space-y-4 text-left">
                <div className="flex gap-4 p-4 bg-slate-50 rounded-2xl border-b-4 border-slate-100">
                  <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-black">
                    1
                  </div>
                  <p className="text-sm font-bold text-muted-foreground">
                    Click any unlocked node to start learning a sign.
                  </p>
                </div>
                <div className="flex gap-4 p-4 bg-slate-50 rounded-2xl border-b-4 border-slate-100">
                  <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-black">
                    2
                  </div>
                  <p className="text-sm font-bold text-muted-foreground">
                    Complete <span className="text-primary">Level 1</span> signs
                    to unlock <span className="text-accent">Level 2</span> and
                    beyond!
                  </p>
                </div>
              </div>

              <GameButton
                variant="duolingo"
                className="w-full py-6 text-2xl"
                onClick={() => setIsHelpOpen(false)}
              >
                GOT IT!
              </GameButton>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
