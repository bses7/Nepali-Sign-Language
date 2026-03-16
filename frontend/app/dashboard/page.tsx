"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { GameButton } from "@/components/game-button";
import { lessonsService } from "@/lib/api/lessons";
import {
  XPBar,
  StreakDisplay,
  LevelCircle,
  Badge,
  CoinDisplay,
  DailyReward,
  WeeklyActivity,
} from "@/components/game-stats";
import { Avatar3DViewer } from "@/components/avatar-viewer-3d";
import { ProfileDropdown } from "@/components/profile-dropdown";
import Link from "next/link";
import { useAuthStore } from "@/lib/store/auth";
import {
  Trophy,
  Star,
  Target,
  Crown,
  Flame,
  ShoppingBag,
  BookOpen,
  ArrowRight,
  GraduationCap,
  MapPin,
} from "lucide-react";
import { cn } from "@/lib/utils";

import { GameShopIcon } from "@/components/icons/game-shop-icon";

interface Lesson {
  name: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  progress: number;
  locked?: boolean;
}

const lessons: Lesson[] = [
  {
    name: "Fingerspelling Basics",
    difficulty: "Beginner",
    progress: 100,
    locked: false,
  },
  {
    name: "Common Phrases",
    difficulty: "Beginner",
    progress: 65,
    locked: false,
  },
  {
    name: "Numbers & Time",
    difficulty: "Intermediate",
    progress: 40,
    locked: false,
  },
  {
    name: "Advanced Conversations",
    difficulty: "Advanced",
    progress: 0,
    locked: false,
  },
];

const achievements = [
  { icon: "🎯", isUnlocked: true },
  { icon: "🔥", isUnlocked: true },
  { icon: "⭐", isUnlocked: true },
  { icon: "👑", isUnlocked: false },
  { icon: "🏆", isUnlocked: false },
  { icon: "🎓", isUnlocked: false },
];

const leaderboardData = [
  { rank: 1, name: "Alex", level: 15, isYou: false },
  { rank: 2, name: "Jordan", level: 14, isYou: false },
  { rank: 3, name: "You", level: 12, isYou: true },
  { rank: 4, name: "Casey", level: 11, isYou: false },
];

export default function Dashboard() {
  const router = useRouter();
  const {
    isAuthenticated,
    user,
    dashboard,
    isDashboardLoading,
    fetchDashboard,
  } = useAuthStore();

  const [signs, setSigns] = useState<any[]>([]);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }
    fetchDashboard();
    lessonsService.getSigns().then((res) => {
      if (res.success) setSigns(res.data || []);
    });
  }, [isAuthenticated, router, fetchDashboard]);

  if (!isAuthenticated || isDashboardLoading) return <LoadingScreen />;

  const displayName = dashboard?.first_name || user?.first_name || "Learner";
  const userStats = {
    level: dashboard?.level || 1,
    currentXP: dashboard?.xp || 0,
    maxXP: (dashboard?.level || 1) * 1000,
    streak: dashboard?.streak_count || 0,
    coins: dashboard?.coins || 0,
    completed: dashboard?.completed_signs || 0,
    total: dashboard?.total_signs || 0,
  };

  const vowels = signs.filter((s) => s.category === "vowel");
  const consonants = signs.filter((s) => s.category === "consonant");

  const getStats = (list: any[]) => {
    // 1. Define the logical order of difficulties
    const difficultyMap: Record<string, number> = {
      easy: 1,
      medium: 2,
      hard: 3,
    };

    // 2. Sort the list: first by Difficulty Level, then by ID
    const orderedList = [...list].sort((a, b) => {
      const diffA = difficultyMap[a.difficulty?.toLowerCase()] || 1;
      const diffB = difficultyMap[b.difficulty?.toLowerCase()] || 1;

      if (diffA !== diffB) {
        return diffA - diffB; // Easy comes before Medium, etc.
      }
      return a.id - b.id; // If difficulty is the same, sort by ID
    });

    const total = orderedList.length || 1;
    const completed = orderedList.filter((s) => s.is_completed).length;
    const percent = Math.round((completed / total) * 100);

    // 3. Find the first sign that is unlocked but NOT completed in the correct order
    const current =
      orderedList.find((s) => !s.is_completed && !s.is_locked) ||
      orderedList[orderedList.length - 1];

    return { total, completed, percent, current };
  };

  const vStats = getStats(vowels);
  const cStats = getStats(consonants);

  return (
    <div className="min-h-screen w-full bg-[#F4EDE4] text-[#2C3E33]">
      {/* 1. GAMIFIED NAVIGATION */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b-4 border-border/50 px-4 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="font-display text-3xl font-black text-primary tracking-tighter">
            SignLearn
          </h1>
          <div className="flex items-center gap-4">
            <CoinDisplay amount={userStats.coins} />
            <Link href="/shop" className="hidden md:block group">
              <button
                className={cn(
                  "relative flex items-center gap-3 px-6 py-2.5 rounded-2xl transition-all duration-75",
                  "bg-white border-b-4 border-slate-200 hover:border-blue-400 active:border-b-0 active:translate-y-1 shadow-lg",
                )}
              >
                <div className="w-8 h-8 group-hover:scale-110 group-hover:rotate-6 transition-transform ">
                  <GameShopIcon />
                </div>

                <div className="flex flex-col items-start leading-none">
                  <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                    Market
                  </span>
                  <span className="text-sm font-black uppercase text-blue-600">
                    Avatar Shop
                  </span>
                </div>

                {/* Small Notification Dot (Optional Gamified Detail) */}
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full border-2 border-white animate-pulse" />
              </button>
            </Link>
            <ProfileDropdown userName={displayName} />
          </div>
        </div>
      </nav>

      <main className="pt-24 pb-12 px-4 max-w-7xl mx-auto space-y-8">
        {/* 2. HERO SECTION */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-primary p-8 rounded-[3rem] text-white shadow-[0_12px_0_0_#4a5f4b] relative overflow-hidden">
              <div className="relative z-10 space-y-4">
                <h1 className="text-5xl font-black uppercase tracking-tighter">
                  Welcome,{" "}
                  <span className="text-yellow-300">{displayName}</span>!
                </h1>
                <p className="font-bold text-primary-foreground/80 max-w-xl text-lg">
                  You're on a {userStats.streak} day streak! Level up to unlock
                  the "Master" avatar.
                </p>
              </div>
              <Star className="absolute -right-10 -bottom-10 w-64 h-64 text-white/10 rotate-12" />
              <div className="mt-6 flex items-center gap-4 bg-black/20 p-4 rounded-3xl border border-white/10 backdrop-blur-sm">
                <div className="w-12 h-12 bg-yellow-400 rounded-2xl flex items-center justify-center text-2xl shadow-lg rotate-3">
                  👑
                </div>
                <div>
                  <p className="text-[10px] font-black uppercase tracking-widest text-yellow-200">
                    Next Unlock at Level {userStats.level + 1}
                  </p>
                  <p className="font-bold text-sm">
                    "Master Signer" Title & Golden Avatar Frame
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Row 1: Experience & Streak */}
              <div className="bg-white p-8 rounded-[2.5rem] border-b-8 border-slate-200 shadow-xl">
                <h3 className="flex items-center gap-2 font-black uppercase text-sm tracking-widest text-muted-foreground mb-4">
                  <Star size={18} className="text-accent fill-accent" /> EXP
                  Progress
                </h3>
                <XPBar
                  current={userStats.currentXP}
                  max={userStats.maxXP}
                  level={userStats.level}
                />
              </div>

              <div className="bg-white p-8 rounded-[2.5rem] border-b-8 border-slate-200 shadow-xl">
                <h3 className="flex items-center gap-2 font-black uppercase text-sm tracking-widest text-muted-foreground mb-4">
                  <Flame
                    size={18}
                    className="text-orange-500 fill-orange-500"
                  />{" "}
                  Daily Fire
                </h3>
                <StreakDisplay
                  count={userStats.streak}
                  highest={userStats.streak}
                />
              </div>

              {/* Row 2: NEW GAMIFIED WIDGETS (Fills the empty space) */}
              <DailyReward canClaim={dashboard?.can_claim_daily} />

              <WeeklyActivity activityDays={dashboard?.weekly_activity || []} />
            </div>
          </div>

          {/* 3. RIGHT SIDE - AVATAR BOX */}
          <div className="bg-white rounded-[3rem] p-6 border-b-[12px] border-slate-200 shadow-2xl space-y-2">
            {/* The 3D Container */}
            <div className="relative h-[560px] w-full rounded-[2.5rem] overflow-hidden bg-slate-900 border-4 border-slate-100 shadow-inner">
              {/* LEVEL BADGE HUD - Positioned Top Left */}
              <div className="absolute top-6 left-6 z-10 pointer-events-none flex flex-col items-center">
                {/* THE CIRCLE: Size remains "md" exactly as before */}
                <LevelCircle
                  level={userStats.level}
                  size="md"
                  variant="success"
                  className="shadow-2xl border-white"
                />
              </div>

              {/* THE 3D VIEWER */}
              <Avatar3DViewer
                avatarFolder={dashboard?.equipped_avatar_folder || "avatar"}
                animationName="BreathingIdle"
              />

              {/* Subtle dark gradient for depth */}
              <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-black/40 to-transparent pointer-events-none" />
            </div>

            {/* CHARACTER INFO & CHANGE BUTTON */}
            <div className="space-y-4">
              <Link href="/shop" className="block">
                <GameButton variant="retro" className="w-full py-2 text-lg">
                  Change Character
                </GameButton>
              </Link>
            </div>
          </div>
        </section>

        {/* 4. MAIN CONTENT GRID */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-8">
            {/* Continue Learning */}
            <div className="bg-white rounded-[3rem] p-8 border-b-[12px] border-slate-200 shadow-xl space-y-8">
              <div className="flex items-center justify-between">
                <h2 className="text-3xl font-black uppercase tracking-tighter flex items-center gap-3">
                  <BookOpen className="text-primary" /> Active Quests
                </h2>
                <div className="bg-primary/10 px-4 py-1 rounded-full border-2 border-primary/20">
                  <span className="text-primary font-black text-xs uppercase tracking-widest">
                    {dashboard?.progress_percentage}% Mastery
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* CARD 1: Vowel Progress (Math) */}
                <LessonProgressCard
                  title="Vowel"
                  icon={<GraduationCap className="text-primary" />}
                  stats={vStats}
                  color="bg-primary"
                  category="vowel"
                />

                {/* CARD 2: Consonant Progress (Math) */}
                <LessonProgressCard
                  title="Consonant"
                  icon={<BookOpen className="text-secondary" />}
                  stats={cStats}
                  color="bg-secondary"
                  category="consonant"
                />

                {/* CARD 3: Current Vowel Marker (Visual Location) */}
                <CurrentMarkerCard
                  title="Vowel"
                  sign={vStats.current}
                  category="vowel"
                />

                {/* CARD 4: Current Consonant Marker (Visual Location) */}
                <CurrentMarkerCard
                  title="Consonant"
                  sign={cStats.current}
                  category="consonant"
                />
              </div>

              <Link href="/lessons" className="block">
                <GameButton variant="duolingo" className="w-full py-2 text-xl">
                  Explore World Map
                </GameButton>
              </Link>
            </div>

            {/* Daily Challenge */}
            <div className="bg-gradient-to-br from-secondary to-accent p-8 rounded-[3rem] text-white shadow-[0_12px_0_0_#829480] space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-3xl font-black uppercase tracking-tighter">
                  Daily Challenge
                </h2>
                <Target size={40} className="animate-pulse" />
              </div>
              <p className="font-bold opacity-90">
                Finish 5 lessons today for a massive +500 XP bonus!
              </p>
              <div className="flex gap-3">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div
                    key={i}
                    className={cn(
                      "h-4 flex-1 rounded-full border-2 border-white/20",
                      i <= userStats.completed % 5
                        ? "bg-yellow-300 shadow-[0_0_15px_rgba(253,224,71,0.5)]"
                        : "bg-white/10",
                    )}
                  />
                ))}
              </div>
              <GameButton variant="retro" className="w-full">
                Start Challenge
              </GameButton>
            </div>
          </div>

          {/* 5. RIGHT COLUMN - ACHIEVEMENTS & LEADERBOARD */}
          <div className="space-y-8">
            {/* Trophy Room (Achievements) Preview Card */}
            <Link href="/shop?tab=badges" className="block group">
              <div className="bg-white rounded-[3rem] p-8 border-b-8 border-slate-200 shadow-xl space-y-6 transition-all hover:border-warning hover:-translate-y-1 active:translate-y-0 active:border-b-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-black uppercase tracking-tight flex items-center gap-2">
                    <Trophy className="text-warning fill-warning group-hover:animate-bounce" />
                    Trophy Room
                  </h3>
                  <span className="text-xs font-black text-warning group-hover:underline">
                    View Badges
                  </span>
                </div>

                {/* Mini Grid of Badges */}
                <div className="grid grid-cols-3 gap-3">
                  {achievements.slice(0, 6).map((achievement, i) => (
                    <div
                      key={i}
                      className={cn(
                        "relative aspect-square rounded-2xl flex items-center justify-center text-2xl border-2 transition-all group-hover:scale-105",
                        achievement.isUnlocked
                          ? "bg-yellow-50 border-yellow-200 shadow-[inset_0_-4px_0_0_#fde047]"
                          : "bg-slate-50 border-slate-100 opacity-30 grayscale",
                      )}
                    >
                      {achievement.icon}
                      {!achievement.isUnlocked && (
                        <div className="absolute inset-0 flex items-center justify-center text-xs">
                          🔒
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Progress Summary */}
                <div className="bg-slate-50 rounded-2xl p-3 border-2 border-slate-100">
                  <div className="flex justify-between text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-1">
                    <span>Collection Progress</span>
                    <span>
                      {achievements.filter((a) => a.isUnlocked).length} /{" "}
                      {achievements.length}
                    </span>
                  </div>
                  <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-warning transition-all duration-1000"
                      style={{
                        width: `${(achievements.filter((a) => a.isUnlocked).length / achievements.length) * 100}%`,
                      }}
                    />
                  </div>
                </div>

                {/* The Visual CTA Button */}
                <div className="pt-1">
                  <div className="w-full bg-warning text-yellow-900 py-3 rounded-2xl font-black uppercase text-center border-b-4 border-yellow-600 group-hover:bg-yellow-400 transition-colors">
                    All Achievements
                  </div>
                </div>
              </div>
            </Link>

            {/* Leaderboard Preview Card */}
            <Link href="/leaderboard" className="block group">
              <div className="bg-white rounded-[3rem] p-8 border-b-8 border-slate-200 shadow-xl space-y-6 transition-all hover:border-accent hover:-translate-y-1 active:translate-y-0 active:border-b-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-black uppercase tracking-tight flex items-center gap-2">
                    <Crown className="text-accent fill-accent group-hover:animate-bounce" />
                    Top Rangers
                  </h3>
                  <span className="text-xs font-black text-primary group-hover:underline">
                    View All
                  </span>
                </div>

                <div className="space-y-3">
                  {[
                    { rank: 1, name: "SignPro", level: 25, icon: "🥇" },
                    { rank: 2, name: "FastLearner", level: 22, icon: "🥈" },
                    {
                      rank: 3,
                      name: displayName,
                      level: userStats.level,
                      icon: "🥉",
                      isYou: true,
                    },
                  ].map((player) => (
                    <div
                      key={player.rank}
                      className={cn(
                        "flex items-center justify-between p-4 rounded-2xl border-2 transition-all",
                        player.isYou
                          ? "bg-primary/10 border-primary"
                          : "bg-slate-50 border-transparent group-hover:bg-slate-100",
                      )}
                    >
                      <div className="flex items-center gap-3">
                        <span className="font-black text-sm w-6">
                          {player.rank}
                        </span>
                        <span
                          className={cn(
                            "font-bold",
                            player.isYou && "text-primary",
                          )}
                        >
                          {player.name} {player.isYou && "⭐"}
                        </span>
                      </div>
                      <span className="font-black text-xs text-primary">
                        LVL {player.level}
                      </span>
                    </div>
                  ))}
                </div>

                {/* The Visual Button - It's inside the link so it's decorative but functional */}
                <div className="pt-1">
                  <div className="w-full bg-warning text-yellow-900 py-3 rounded-2xl font-black uppercase text-center border-b-4 border-yellow-600 group-hover:bg-yellow-400 transition-colors">
                    Full Leaderboard
                  </div>
                </div>
              </div>
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}

function LessonProgressCard({ title, icon, stats, color, category }: any) {
  return (
    <Link href={`/lessons?category=${category}`} className="group block">
      <div className="bg-slate-50 rounded-[2.5rem] p-6 border-b-8 border-slate-200 hover:border-primary/40 transition-all space-y-4">
        <div className="flex justify-between items-center">
          <div className="p-3 bg-white rounded-2xl border-b-4 border-slate-100">
            {icon}
          </div>
          <p className="text-2xl font-black text-foreground">
            {stats.percent}%
          </p>
        </div>
        <div>
          <h4 className="font-black uppercase text-sm tracking-widest text-muted-foreground">
            {title}
          </h4>
          <div className="mt-2 w-full h-3 bg-slate-200 rounded-full overflow-hidden border border-slate-300/30">
            <div
              className={cn("h-full transition-all duration-1000", color)}
              style={{ width: `${stats.percent}%` }}
            />
          </div>
          <p className="text-[10px] font-bold text-muted-foreground mt-2 uppercase tracking-tighter">
            {stats.completed} of {stats.total} Signs Learned
          </p>
        </div>
      </div>
    </Link>
  );
}

// --- SUB-COMPONENT: MARKER CARDS (LOCATION-BASED) ---
function CurrentMarkerCard({ title, sign, category }: any) {
  return (
    <Link href={`/lessons/${sign?.id}`} className="group block">
      <div className="bg-slate-50 rounded-[2.5rem] p-6 border-b-8 border-slate-200 hover:border-accent transition-all flex flex-col justify-between min-h-[160px]">
        <div className="flex justify-between items-start">
          <MapPin className="text-accent" size={24} />

          <div className="text-right">
            <p className="text-[9px] font-black uppercase text-muted-foreground tracking-widest leading-none">
              Continue Learning
            </p>
            <p className="text-xs font-black uppercase text-accent mt-1">
              {title}
            </p>
          </div>
        </div>

        <div className="bg-white p-3 rounded-2xl border-2 border-slate-100 flex items-center justify-between group-hover:bg-slate-50 transition-colors">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-slate-900 rounded-xl flex items-center justify-center text-white font-black text-2xl shadow-lg ring-2 ring-white">
              {sign?.nepali_char || "✓"}
            </div>
            <div>
              <p className="text-[10px] font-black uppercase text-muted-foreground leading-none">
                Next Sign
              </p>
              <h5 className="font-black uppercase text-lg leading-tight tracking-tighter">
                {sign?.title || "Mastered!"}
              </h5>
            </div>
          </div>
          <div className="bg-[#ff9600] p-2 rounded-xl text-white border-b-4 border-[#d97900]group-hover:translate-x-1 transition-all">
            <ArrowRight size={18} />
          </div>
        </div>
      </div>
    </Link>
  );
}
function LoadingScreen() {
  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background gap-4">
      <div className="w-16 h-16 border-8 border-primary/20 border-t-primary rounded-full animate-spin" />
      <p className="font-black uppercase tracking-widest text-primary animate-pulse">
        Syncing Player Data...
      </p>
    </div>
  );
}
