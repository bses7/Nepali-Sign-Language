"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Flame, Trophy, Star } from "lucide-react";
import { GameButton } from "@/components/game-button";
import { useRouter } from "next/navigation";

interface XPBarProps {
  current: number;
  max: number;
  level: number;
  className?: string;
  showLabel?: boolean;
}

export const XPBar: React.FC<XPBarProps> = ({ current, max, level }) => {
  const percentage = (current / max) * 100;
  return (
    <div className="w-full">
      <div className="flex justify-between text-[10px] font-black uppercase tracking-[0.2em] mb-2 text-muted-foreground">
        <span>Level {level}</span>
        <span>
          {current} / {max} XP
        </span>
      </div>
      <div className="relative w-full h-10 bg-slate-100 border-b-4 border-slate-200 rounded-2xl overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-primary to-[#8dc63f] transition-all duration-1000 shadow-[inset_0_4px_0_rgba(255,255,255,0.3)]"
          style={{ width: `${percentage}%` }}
        />
        {/* Animated Gloss Effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-[shimmer_3s_infinite]" />
      </div>
    </div>
  );
};

interface StreakDisplayProps {
  count: number;
  highest: number;
  className?: string;
}

export const StreakDisplay: React.FC<StreakDisplayProps> = ({ count }) => {
  return (
    <div className="relative group">
      <div className="absolute -inset-1 bg-gradient-to-r from-orange-500 to-yellow-500 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
      <div className="relative flex items-center gap-4 bg-white dark:bg-slate-900 border-b-4 border-orange-200 dark:border-orange-900 p-4 rounded-2xl">
        <div className="bg-orange-500 p-3 rounded-xl">
          <Flame size={24} className="text-white fill-white" />
        </div>
        <div>
          <p className="text-[10px] font-black uppercase tracking-widest text-orange-500">
            Day Streak
          </p>
          <p className="text-3xl font-black leading-none">{count}</p>
        </div>
      </div>
    </div>
  );
};

interface LevelCircleProps {
  level: number;
  size?: "sm" | "md" | "lg";
  variant?: "default" | "success" | "warning" | "accent";
  className?: string;
}

export const LevelCircle: React.FC<LevelCircleProps> = ({
  level,
  size = "md",
}) => {
  const sizes = { sm: "w-12 h-12", md: "w-20 h-20", lg: "w-28 h-28" };

  return (
    <div className={cn("relative group", sizes[size])}>
      <div className="absolute inset-0 bg-primary/20 rounded-full animate-spin-slow" />

      <div className="absolute inset-0 flex flex-col items-center justify-center bg-primary border-b-8 border-primary-foreground/30 rounded-full shadow-xl">
        <span className="text-white font-black text-2xl leading-none drop-shadow-md">
          {level}
        </span>

        <span className="text-white/80 font-black text-[8px] uppercase tracking-[0.2em] leading-none mt-1">
          LVL
        </span>
      </div>

      <div className="absolute -top-1 -right-1 bg-yellow-400 p-1.5 rounded-lg border-2 border-white rotate-12 group-hover:rotate-0 transition-transform">
        <Star size={14} className="fill-white text-white" />
      </div>
    </div>
  );
};

interface BadgeProps {
  title: string;
  description?: string;
  icon: React.ReactNode;
  isUnlocked?: boolean;
  rarity?: "common" | "rare" | "epic" | "legendary";
}

export const Badge: React.FC<BadgeProps> = ({
  title,
  description,
  icon,
  isUnlocked = true,
  rarity = "common",
}) => {
  const rarityClasses = {
    common: "border-gray-400 bg-gray-100 dark:bg-gray-800",
    rare: "border-blue-500 bg-blue-100 dark:bg-blue-900",
    epic: "border-purple-500 bg-purple-100 dark:bg-purple-900",
    legendary: "border-yellow-500 bg-yellow-100 dark:bg-yellow-900",
  };

  return (
    <div
      className={cn(
        "relative w-24 h-24 rounded-full flex flex-col items-center justify-center border-4 cursor-pointer transition-transform hover:scale-110",
        isUnlocked
          ? rarityClasses[rarity]
          : "bg-muted border-muted-foreground opacity-50",
        isUnlocked && "badge-pulse",
      )}
      title={`${title}${description ? ": " + description : ""}`}
    >
      {!isUnlocked && (
        <div className="absolute inset-0 rounded-full bg-black/20" />
      )}
      <div className="text-2xl">{icon}</div>
      {!isUnlocked && <div className="absolute text-2xl">🔒</div>}
    </div>
  );
};

interface CoinDisplayProps {
  amount: number;
  className?: string;
}

export const CoinDisplay: React.FC<CoinDisplayProps> = ({
  amount,
  className,
}) => {
  return (
    <div
      className={cn(
        "flex items-center gap-2 bg-card px-4 py-2 rounded-full border-2 border-warning shadow-lg",
        className,
      )}
    >
      <span className="text-2xl">💰</span>
      <span className="font-bold font-display text-lg text-foreground">
        {amount}
      </span>
    </div>
  );
};

interface DailyRewardProps {
  canClaim?: boolean;
}

export const DailyReward: React.FC<DailyRewardProps> = ({
  canClaim = true,
}) => {
  const router = useRouter();

  return (
    <div
      className={cn(
        "bg-white p-6 rounded-[2.5rem] border-b-8 shadow-xl flex items-center justify-between group transition-all",
        canClaim
          ? "border-slate-200 hover:border-primary/30"
          : "border-slate-100 opacity-80",
      )}
    >
      <div className="flex items-center gap-4">
        {/* If claimed, stop the bounce and use a checkmark or open box */}
        <div
          className={cn(
            "text-4xl",
            canClaim ? "animate-bounce" : "grayscale opacity-50",
          )}
        >
          {canClaim ? "🎁" : "📦"}
        </div>
        <div>
          <h4
            className={cn(
              "font-black uppercase text-sm tracking-tight",
              canClaim ? "text-foreground" : "text-muted-foreground",
            )}
          >
            Daily Reward
          </h4>
          <p className="text-[10px] font-black uppercase text-primary tracking-widest">
            {canClaim ? "100 Coins Waiting!" : "See you tomorrow!"}
          </p>
        </div>
      </div>

      <GameButton
        variant={canClaim ? "retro" : "duolingo"}
        size="md"
        onClick={() => router.push("/shop?tab=rewards")}
        className={cn(!canClaim && "grayscale cursor-default")}
      >
        {canClaim ? "Claim" : "Claimed"}
      </GameButton>
    </div>
  );
};

interface WeeklyActivityProps {
  activityDays?: number[]; // Expecting something like [0, 2, 6]
}

export const WeeklyActivity: React.FC<WeeklyActivityProps> = ({
  activityDays = [],
}) => {
  const days = [
    { label: "M", id: 0 },
    { label: "T", id: 1 },
    { label: "W", id: 2 },
    { label: "T", id: 3 },
    { label: "F", id: 4 },
    { label: "S", id: 5 },
    { label: "S", id: 6 },
  ];

  return (
    <div className="bg-white p-6 rounded-[2.5rem] border-b-8 border-slate-200 shadow-xl flex flex-col justify-between">
      <h4 className="font-black uppercase text-[10px] tracking-[0.2em] text-muted-foreground mb-4 text-center">
        Weekly Activity
      </h4>
      <div className="flex justify-between items-center px-1">
        {days.map((day) => {
          const isActive = activityDays.includes(day.id);

          return (
            <div key={day.id} className="flex flex-col items-center gap-2">
              {/* 3D Training Gem */}
              <div
                className={cn(
                  "w-9 h-9 rounded-xl border-b-4 flex items-center justify-center text-xs font-black transition-all duration-300",
                  isActive
                    ? "bg-primary text-white border-[#4a5f4b] shadow-[0_4px_10px_rgba(95,122,97,0.4)] scale-110"
                    : "bg-slate-100 text-slate-300 border-slate-200 opacity-60",
                )}
                title={isActive ? "Training Complete!" : "No activity"}
              >
                {isActive ? "✓" : ""}
              </div>
              <span
                className={cn(
                  "text-[10px] font-black transition-colors",
                  isActive ? "text-primary" : "text-muted-foreground/50",
                )}
              >
                {day.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
