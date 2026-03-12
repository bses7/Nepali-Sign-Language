"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { usersService } from "@/lib/api/users";
import { GameButton } from "@/components/game-button";
import { CoinDisplay } from "@/components/game-stats";
import { ProfileDropdown } from "@/components/profile-dropdown";
import {
  ShoppingBag,
  Crown,
  Flame,
  Trophy,
  ArrowLeft,
  Medal,
  Undo2,
  ChevronLeft,
} from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";

export default function LeaderboardPage() {
  const router = useRouter();
  const { isAuthenticated, user, dashboard, fetchDashboard } = useAuthStore();
  const [leaderboard, setLeaderboard] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }

    const loadData = async () => {
      setIsLoading(true);
      const [lbRes] = await Promise.all([
        usersService.getLeaderboard(),
        fetchDashboard(),
      ]);

      if (lbRes.success && lbRes.data) {
        setLeaderboard((lbRes.data as any).top_users || []);
      }
      setIsLoading(false);
    };

    loadData();
  }, [isAuthenticated, router, fetchDashboard]);

  if (isLoading) return <LoadingScreen />;

  const displayName = dashboard?.first_name || user?.first_name || "Learner";

  return (
    <div className="min-h-screen w-full bg-[#F4EDE4] text-[#2C3E33]">
      {/* 1. DASHBOARD-STYLE NAVIGATION */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b-4 border-border/50 px-4 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <GameButton
              variant="back"
              onClick={() => router.push("/dashboard")}
            >
              <ChevronLeft size={24} strokeWidth={3} />
            </GameButton>
            <h1 className="font-display text-3xl font-black text-primary tracking-tighter">
              SignLearn
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <CoinDisplay amount={dashboard?.coins || 0} />
            <Link href="/shop" className="hidden md:block">
              <button className="flex items-center gap-2 px-4 py-2 rounded-2xl bg-secondary/10 border-b-4 border-secondary text-secondary font-black uppercase text-xs transition-all active:translate-y-1 active:border-b-0">
                <ShoppingBag size={18} />
                <span>Shop</span>
              </button>
            </Link>
            <ProfileDropdown userName={displayName} />
          </div>
        </div>
      </nav>

      <main className="pt-28 pb-12 px-4 max-w-4xl mx-auto space-y-8">
        {/* 2. HERO HEADER */}
        <div className="text-center space-y-4">
          <div className="inline-block bg-yellow-400 p-4 rounded-[2rem] border-b-8 border-yellow-600 animate-bounce mb-2">
            <Trophy size={48} className="text-white fill-white" />
          </div>
          <h1 className="font-display text-5xl font-black uppercase tracking-tighter">
            Global <span className="text-primary">Hall of Fame</span>
          </h1>
          <p className="font-bold text-muted-foreground uppercase tracking-widest text-sm">
            Top Rangers this week
          </p>
        </div>

        {/* 3. THE LEADERBOARD LIST */}
        <div className="bg-white rounded-[3rem] p-4 md:p-8 border-b-[12px] border-slate-200 shadow-2xl space-y-4">
          {leaderboard.map((entry, index) => {
            const rank = index + 1;
            // Recognized the current user by comparing emails from backend
            const isMe = entry.email === user?.email;

            return (
              <div
                key={entry.email || index}
                className={cn(
                  "relative flex items-center justify-between p-5 rounded-[2rem] border-2 transition-all duration-300",
                  isMe
                    ? "bg-primary/10 border-primary border-b-8 -translate-y-1 z-10"
                    : "bg-slate-50 border-transparent hover:bg-slate-100",
                )}
              >
                {/* Left Side: Rank & Name */}
                <div className="flex items-center gap-4 md:gap-6">
                  <div
                    className={cn(
                      "w-12 h-12 rounded-2xl flex items-center justify-center font-black text-xl border-b-4",
                      rank === 1
                        ? "bg-yellow-400 border-yellow-600 text-white"
                        : rank === 2
                          ? "bg-slate-300 border-slate-400 text-slate-600"
                          : rank === 3
                            ? "bg-orange-400 border-orange-600 text-white"
                            : "bg-white border-slate-200 text-muted-foreground",
                    )}
                  >
                    {rank === 1 ? <Crown size={24} /> : rank}
                  </div>

                  <div>
                    <p
                      className={cn(
                        "font-black text-lg md:text-xl uppercase tracking-tight",
                        isMe ? "text-primary" : "text-foreground",
                      )}
                    >
                      {entry.first_name} {entry.last_name} {isMe && "(YOU)"}
                    </p>
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
                        Level {entry.level || 1}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Right Side: Stats */}
                <div className="flex items-center gap-4 md:gap-8">
                  <div className="text-right hidden sm:block">
                    <p className="text-[10px] font-black uppercase text-muted-foreground tracking-widest">
                      XP Points
                    </p>
                    <p className="font-black text-lg text-accent">
                      {(entry.xp || 0).toLocaleString()}
                    </p>
                  </div>

                  {/* Streak Power-up Box */}
                  <div className="flex flex-col items-center bg-white border-2 border-orange-200 px-3 py-1.5 rounded-2xl shadow-[0_4px_0_0_#fb923c]">
                    <Flame
                      size={16}
                      className="text-orange-500 fill-orange-500"
                    />
                    <span className="font-black text-orange-500 text-sm">
                      {entry.streak_count || 0}
                    </span>
                  </div>
                </div>

                {/* Winner Badge for Rank 1 */}
                {rank === 1 && (
                  <div className="absolute -top-3 -right-3 bg-yellow-400 text-white text-[10px] font-black px-3 py-1 rounded-full border-2 border-white rotate-12 shadow-lg">
                    TOP RANGER
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* 4. FOOTER ACTION */}
        <div className="text-center pt-4 pb-8">
          <Link href="/dashboard">
            <GameButton variant="duolingo" size="lg" className="px-12">
              Back to Quests
            </GameButton>
          </Link>
        </div>
      </main>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-[#F4EDE4] gap-4">
      <div className="w-16 h-16 border-8 border-primary/20 border-t-primary rounded-full animate-spin" />
      <p className="font-black uppercase tracking-widest text-primary animate-pulse">
        Calculating Ranks...
      </p>
    </div>
  );
}
