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
  Lock,
  Sparkles,
  BadgeCheck,
} from "lucide-react";
import { cn } from "@/lib/utils";

import { GameShopIcon } from "@/components/icons/game-shop-icon";
import { usersService } from "@/lib/api/users";
import { toast } from "sonner";

interface Lesson {
  name: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  progress: number;
  locked?: boolean;
}

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
  const [badges, setBadges] = useState<any[]>([]);
  const [leaderboard, setLeaderboard] = useState<any[]>([]);

  const [showChallengeError, setShowChallengeError] = useState(false);
  const [isClaimingChallenge, setIsClaimingChallenge] = useState(false);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }
    fetchDashboard();
    lessonsService.getSigns().then((res) => {
      if (res.success) setSigns(res.data || []);
    });
    usersService.getAllBadges().then((res) => {
      if (res.success && res.data) {
        const sorted = [...res.data].sort((a, b) => {
          if (a.is_earned === b.is_earned) return 0;
          return a.is_earned ? -1 : 1;
        });
        setBadges(sorted);
      }
    });
    usersService.getLeaderboard().then((res) => {
      if (res.success && res.data) {
        setLeaderboard((res.data as any).top_users || []);
      }
    });
  }, [isAuthenticated, router, fetchDashboard]);

  const handleClaimChallenge = async () => {
    if (isClaimingChallenge) return;

    if (!dashboard?.can_claim_challenge) {
      setShowChallengeError(true);
      setTimeout(() => setShowChallengeError(false), 3000);
      return;
    }

    setIsClaimingChallenge(true);
    try {
      const res = await usersService.claimChallengeReward();
      if (res.success) {
        toast.success("Challenge Reward Claimed!", {
          description: "+250 Coins and +500XP earned!",
          icon: "🎁",
        });
        fetchDashboard();
      } else {
        toast.error(res.error || "Failed to claim reward");
      }
    } finally {
      setIsClaimingChallenge(false);
    }
  };

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
    const difficultyMap: Record<string, number> = {
      easy: 1,
      medium: 2,
      hard: 3,
    };

    const orderedList = [...list].sort((a, b) => {
      const diffA = difficultyMap[a.difficulty?.toLowerCase()] || 1;
      const diffB = difficultyMap[b.difficulty?.toLowerCase()] || 1;

      if (diffA !== diffB) {
        return diffA - diffB;
      }
      return a.id - b.id;
    });

    const total = orderedList.length || 1;
    const completed = orderedList.filter((s) => s.is_completed).length;
    const percent = Math.round((completed / total) * 100);

    const current =
      orderedList.find((s) => !s.is_completed && !s.is_locked) ||
      orderedList[orderedList.length - 1];

    return { total, completed, percent, current };
  };

  const vStats = getStats(vowels);
  const cStats = getStats(consonants);

  const findUserIndex = (list: any[]) => {
    return list.findIndex(
      (entry) =>
        entry.email === user?.email ||
        (entry.first_name === user?.first_name &&
          entry.last_name === user?.last_name),
    );
  };

  const myIndex = findUserIndex(leaderboard);
  const myRankNum = myIndex !== -1 ? myIndex + 1 : 999;

  const rank1 = leaderboard[0];
  const rank2 = leaderboard[1];

  const isRankMe = (index: number) => index === myIndex;

  const nextBadge = badges.find((b) => !b.is_earned);

  return (
    <div className="min-h-screen w-full bg-[#F4EDE4] text-[#2C3E33]">
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

                <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full border-2 border-white animate-pulse" />
              </button>
            </Link>
            <ProfileDropdown userName={displayName} />

            {dashboard?.is_verified_teacher && (
              <div className="relative group cursor-help">
                <div className="bg-blue-500 rounded-3xl duration-500">
                  <BadgeCheck size={32} className="text-white " />
                </div>

                {/* Tooltip on Hover */}
                <div className="absolute top-14 right-0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                  <div className="bg-slate-900 text-white text-[10px] font-black uppercase tracking-widest px-3 py-1.5 rounded-lg border-b-4 border-black whitespace-nowrap">
                    Verified Instructor
                  </div>
                  <div className="w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-b-[5px] border-b-slate-900 mx-auto -mt-[22px] rotate-180 mb-4 ml-4" />
                </div>
              </div>
            )}
          </div>
        </div>
      </nav>

      <main className="pt-24 pb-12 px-4 max-w-7xl mx-auto space-y-8">
        {/* 2. HERO SECTION */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-primary p-8 rounded-[3rem] text-white shadow-[0_12px_0_0_#4a5f4b] relative overflow-hidden">
              <div className="relative z-10 space-y-4">
                <h1 className="text-5xl font-black uppercase tracking-tighter text-shadow-sm">
                  Welcome,{" "}
                  <span className="text-yellow-300">{displayName}</span>!
                </h1>
                <p className="font-bold text-primary-foreground/80 max-w-xl text-lg leading-relaxed">
                  You're on a {userStats.streak} day streak! Level up and claim
                  coins to unlock the "Legendary" avatars.
                </p>
              </div>

              <Star className="absolute -right-10 -bottom-10 w-64 h-64 text-white/10 rotate-12" />

              <div className="mt-6 flex items-center gap-4 bg-black/20 p-4 rounded-3xl border border-white/10 backdrop-blur-sm relative z-10">
                {nextBadge ? (
                  <>
                    <div className="relative group/badge">
                      <div className="w-14 h-14 bg-white/10 rounded-2xl flex items-center justify-center border-b-4 border-white/20 shadow-lg overflow-hidden backdrop-blur-md">
                        <img
                          src={
                            nextBadge.icon_url.startsWith("http")
                              ? nextBadge.icon_url
                              : `http://localhost:8000${nextBadge.icon_url}`
                          }
                          alt="Next Badge"
                          className="w-10 h-10 object-contain grayscale brightness-125 opacity-80"
                          onError={(e) => {
                            (e.target as HTMLImageElement).src =
                              "https://ui-avatars.com/api/?name=?";
                          }}
                        />
                      </div>
                    </div>

                    <div className="flex-1">
                      <p className="text-[10px] font-black uppercase tracking-[0.2em] text-yellow-300 leading-none mb-1">
                        Earn Your Next Badge
                      </p>
                      <h4 className="font-black uppercase text-sm tracking-tight">
                        {nextBadge.description}
                      </h4>
                      <p className="text-[11px] font-bold text-white/60 line-clamp-1">
                        {nextBadge.name}
                      </p>
                    </div>
                  </>
                ) : (
                  /* Fallback if all badges are earned */
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 bg-yellow-400 rounded-2xl flex items-center justify-center text-2xl shadow-lg rotate-3">
                      🏆
                    </div>
                    <div>
                      <p className="text-[10px] font-black uppercase tracking-widest text-yellow-200">
                        Achievement Maxed!
                      </p>
                      <p className="font-bold text-sm">
                        You have collected all legendary badges!
                      </p>
                    </div>
                  </div>
                )}
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
                  min={userStats.maxXP - 1000}
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
                animationName="Idle"
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
            <div className="bg-gradient-to-br from-secondary to-accent p-8 rounded-[3rem] text-white shadow-[0_12px_0_0_#829480] space-y-6 relative">
              <div className="flex justify-between items-center">
                <div className="space-y-1">
                  <h2 className="text-3xl font-black uppercase tracking-tighter">
                    Daily Challenge
                  </h2>
                  <p className="font-bold opacity-90 text-sm">
                    {dashboard?.challenge_description}
                  </p>
                </div>
                <Target
                  size={44}
                  className={cn(
                    "transition-transform duration-700",
                    dashboard?.can_claim_challenge
                      ? "animate-bounce text-yellow-300"
                      : "opacity-50",
                  )}
                />
              </div>

              {/* DYNAMIC PROGRESS BARS */}
              <div className="flex gap-2">
                {[...Array(dashboard?.challenge_target)].map((_, i) => (
                  <div
                    key={i}
                    className={cn(
                      "h-4 flex-1 rounded-full border-2 transition-all duration-700",
                      i < (dashboard?.challenge_progress || 0)
                        ? "bg-[#b5e61d] border-[#235c24] shadow-[0_0_15px_rgba(253,224,71,0.5)]"
                        : "bg-white/10 border-white/20",
                    )}
                  />
                ))}
              </div>

              <p className="text-[10px] font-black uppercase tracking-widest text-white/70">
                Progress: {dashboard?.challenge_progress} /{" "}
                {dashboard?.challenge_target} Completed
              </p>

              {/* ACTION BUTTON WITH POPUP ERROR */}
              <div className="relative pt-2">
                {showChallengeError && (
                  <div className="absolute -top-14 left-1/2 -translate-x-1/2 w-full animate-pop-spin z-20 pointer-events-none">
                    <div className="bg-red-500 text-white px-4 py-2 rounded-xl border-b-4 border-red-800 shadow-xl text-center">
                      <span className="text-[10px] font-black uppercase tracking-widest">
                        Finish the quest first! ⚔️
                      </span>
                    </div>
                    <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[8px] border-t-red-500 mx-auto" />
                  </div>
                )}

                <GameButton
                  variant={dashboard?.can_claim_challenge ? "jelly" : "retro"}
                  className={cn(
                    "w-full py-2 text-xl",
                    !dashboard?.can_claim_challenge && "opacity-80",
                  )}
                  onClick={handleClaimChallenge}
                  isLoading={isClaimingChallenge}
                >
                  {dashboard?.can_claim_challenge
                    ? "CLAIM CHALLENGE REWARD 🎁"
                    : "CLAIM CHALLENGE REWARD"}
                </GameButton>
              </div>
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
                    View All
                  </span>
                </div>

                {/* Sorted Grid: Showing first 6 slots */}
                <div className="grid grid-cols-3 gap-5">
                  {(badges.length > 0
                    ? badges.slice(0, 6)
                    : Array(6).fill(null)
                  ).map((badge, i) => {
                    if (!badge)
                      return (
                        <div
                          key={i}
                          className="aspect-square rounded-2xl bg-slate-50 border-2 border-slate-100"
                        />
                      );

                    const isEarned = badge.is_earned;
                    const iconUrl = badge.icon_url.startsWith("http")
                      ? badge.icon_url
                      : `http://localhost:8000${badge.icon_url}`;

                    return (
                      <div
                        key={badge.id}
                        className={cn(
                          "relative aspect-square rounded-2xl flex items-center justify-center p-2 border-2 transition-all group-hover:scale-110 overflow-hidden",
                          isEarned
                            ? "bg-yellow-50 border-yellow-200 shadow-[inset_0_-4px_0_0_#fde047]"
                            : "bg-slate-50 border-slate-100 opacity-30 grayscale",
                        )}
                      >
                        <img
                          src={iconUrl}
                          alt={badge.name}
                          className="w-8/12 h-8/12 object-contain"
                          onError={(e) => {
                            (e.target as HTMLImageElement).src =
                              "https://ui-avatars.com/api/?name=" + badge.name;
                          }}
                        />
                        {/* Lock overlay for unearned badges */}
                        {!isEarned && (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <Lock size={10} className="text-slate-400" />
                          </div>
                        )}
                        {/* Sparkle for earned badges */}
                        {isEarned && (
                          <div className="absolute top-0.5 right-0.5">
                            <Sparkles
                              size={10}
                              className="text-yellow-500 animate-pulse"
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Progress Tracking Section */}
                <div className="bg-slate-50 rounded-2xl p-4 border-2 border-slate-100">
                  <div className="flex justify-between text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2">
                    <span>Badge Progress</span>
                    <span>
                      {badges.filter((b) => b.is_earned).length} /{" "}
                      {badges.length || 0}
                    </span>
                  </div>
                  <div className="h-2.5 w-full bg-slate-200 rounded-full overflow-hidden shadow-inner">
                    <div
                      className="h-full bg-warning transition-all duration-1000 shadow-[0_0_10px_#fde047]"
                      style={{
                        width: `${badges.length ? (badges.filter((b) => b.is_earned).length / badges.length) * 100 : 0}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="pt-1">
                  <div className="w-full bg-warning text-white py-4 rounded-2xl font-black uppercase text-center border-b-4 border-yellow-600 group-hover:bg-yellow-400 transition-colors">
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
                    <Crown className="text-accent fill-accent group-hover:animate-bounce" />{" "}
                    Rankings
                  </h3>
                  <span className="text-xs font-black text-primary group-hover:underline">
                    Hall of Fame
                  </span>
                </div>

                <div className="space-y-3">
                  {/* RANK 1 SLOT */}
                  {rank1 && (
                    <RankItem
                      rank={1}
                      name={isRankMe(0) ? "You" : rank1.first_name}
                      level={rank1.level}
                      isMe={isRankMe(0)}
                      isSpecial
                    />
                  )}

                  {/* RANK 2 SLOT */}
                  {rank2 && (
                    <RankItem
                      rank={2}
                      name={isRankMe(1) ? "You" : rank2.first_name}
                      level={rank2.level}
                      isMe={isRankMe(1)}
                    />
                  )}

                  {/* SHOW "YOU" ONLY IF YOU ARE RANK 3 OR LOWER */}
                  {myRankNum > 2 && (
                    <>
                      <div className="flex justify-center gap-1 opacity-20 py-1">
                        <div className="w-1 h-1 bg-black rounded-full" />
                        <div className="w-1 h-1 bg-black rounded-full" />
                        <div className="w-1 h-1 bg-black rounded-full" />
                      </div>
                      <div className="flex items-center justify-between p-4 rounded-2xl border-2 border-primary bg-primary/10 shadow-md">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-primary rounded-xl flex items-center justify-center text-white border-b-4 border-[#4a5f4b] font-black text-xs">
                            {myRankNum}
                          </div>
                          <span className="font-black text-sm text-primary uppercase tracking-tight">
                            You
                          </span>
                        </div>
                        <span className="font-black text-xs text-primary uppercase">
                          LVL {dashboard?.level}
                        </span>
                      </div>
                    </>
                  )}
                </div>

                <div className="pt-1">
                  <div className="w-full bg-warning text-white py-4 rounded-2xl font-black uppercase text-center border-b-4 border-yellow-600 group-hover:bg-yellow-400 transition-colors">
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

function RankItem({ rank, name, level, isSpecial = false }: any) {
  return (
    <div
      className={cn(
        "flex items-center justify-between p-4 rounded-2xl border-2",
        isSpecial
          ? "border-yellow-200 bg-yellow-50/50"
          : "border-slate-50 bg-slate-50/30",
      )}
    >
      <div className="flex items-center gap-3">
        <div
          className={cn(
            "w-10 h-10 rounded-xl flex items-center justify-center border-b-4 text-xs font-black",
            isSpecial
              ? "bg-yellow-400 border-yellow-600 text-white"
              : "bg-slate-200 border-slate-300 text-slate-500",
          )}
        >
          {isSpecial ? <Crown size={18} /> : rank}
        </div>
        <span className="font-bold text-sm truncate max-w-[100px]">{name}</span>
      </div>
      <span
        className={cn(
          "font-black text-[10px] uppercase",
          isSpecial ? "text-yellow-600" : "text-slate-400",
        )}
      >
        LVL {level}
      </span>
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
