"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { GameButton } from "@/components/game-button";
import { CoinDisplay } from "@/components/game-stats";
import { ProfileDropdown } from "@/components/profile-dropdown";
import { Avatar3DViewer } from "@/components/avatar-viewer-3d";
import { useAuthStore } from "@/lib/store/auth";
import { usersService } from "@/lib/api/users";
import { avatarService, AvatarItem } from "@/lib/api/avatar";
import {
  ShoppingBag,
  Sparkles,
  Zap,
  Info,
  Star,
  ChevronLeft,
  HelpCircle,
  Flame,
  BookOpen,
  Gift,
  Trophy,
  CheckCircle2,
  ChevronUp,
  Lock,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import Link from "next/link";
import { GameChest3D } from "@/components/game-chest-3d";

function ShopContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const initialTab =
    (searchParams.get("tab") as "avatars" | "badges" | "rewards") || "avatars";
  const [selectedTab, setSelectedTab] = useState<
    "avatars" | "badges" | "rewards"
  >(initialTab);

  const {
    isAuthenticated,
    user,
    dashboard,
    fetchDashboard,
    isDashboardLoading,
  } = useAuthStore();

  const [avatars, setAvatars] = useState<AvatarItem[]>([]);
  const [ownedBadges, setOwnedBadges] = useState<string[]>([]);
  const [isClaiming, setIsClaiming] = useState(false);
  const [chestOpen, setChestOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  function StatBar({
    label,
    value,
    color,
  }: {
    label: string;
    value: number;
    color: string;
  }) {
    return (
      <div className="space-y-1">
        <div className="flex justify-between text-[10px] font-black uppercase">
          <span>{label}</span>
          <span className="text-muted-foreground">{value}%</span>
        </div>
        <div className="h-2 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-1000",
              color,
            )}
            style={{ width: `${value}%` }}
          />
        </div>
      </div>
    );
  }

  const tabHeaderInfo = {
    avatars: {
      badge: "Trading Post",
      main: "Avatar",
      sub: "Shop",
      desc: "Spend your hard-earned coins on legendary gear and profile enhancements.",
    },
    badges: {
      badge: "Hall of Fame",
      main: "Your",
      sub: "Collection",
      desc: "View your legendary achievements and certificates of mastery.",
    },
    rewards: {
      badge: "The Vault",
      main: "Daily",
      sub: "Loot",
      desc: "Unlock legendary rewards and claim your daily treasure.",
    },
  };

  useEffect(() => {
    if (dashboard) setChestOpen(!dashboard.can_claim_daily);
  }, [dashboard]);

  useEffect(() => {
    const tab = searchParams.get("tab") as any;
    if (tab === "avatars" || tab === "badges" || tab === "rewards")
      setSelectedTab(tab);
  }, [searchParams]);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }
    fetchDashboard();

    avatarService.getAvatarStore().then((res) => {
      if (res.success) setAvatars(res.data || []);
    });

    usersService.getUserBadges().then((res) => {
      if (res.success)
        setOwnedBadges(res.data?.map((b: any) => b.badge_code) || []);
    });
  }, [isAuthenticated, router, fetchDashboard]);

  const handleAvatarAction = async (avatar: AvatarItem) => {
    setIsProcessing(true);
    try {
      if (avatar.is_owned) {
        const res = await avatarService.equipAvatar(avatar.id);
        if (res.success) {
          toast.success(`${avatar.name} equipped!`);
          fetchDashboard();
        }
      } else {
        if ((dashboard?.coins || 0) < avatar.price) {
          toast.error("Not enough coins!");
          return;
        }
        const res = await avatarService.buyAvatar(avatar.id);
        if (res.success) {
          toast.success(`Purchased ${avatar.name}!`);
          const storeRes = await avatarService.getAvatarStore();
          if (storeRes.success) setAvatars(storeRes.data || []);
          fetchDashboard();
        }
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClaimDaily = async () => {
    if (chestOpen || isClaiming || !dashboard?.can_claim_daily) return;
    setIsClaiming(true);
    try {
      const res = await usersService.claimDailyReward();
      if (res.success) {
        setChestOpen(true);
        toast.success("Successfully claimed 100 coins!");
        fetchDashboard();
      }
    } finally {
      setIsClaiming(false);
    }
  };

  if (isDashboardLoading) return <LoadingScreen />;

  const userCoins = dashboard?.coins ?? 0;
  const displayName = dashboard?.first_name || user?.first_name || "Learner";

  const backendBadges = [
    {
      name: "Vowel Master",
      code: "VOWEL_MASTER",
      desc: "Learned all 13 Nepali vowels.",
      icon: "👄",
    },
    {
      name: "Early Bird",
      code: "EARLY_BIRD",
      desc: "Completed a lesson before 7 AM.",
      icon: "🌅",
    },
    {
      name: "Consistent Learner",
      code: "CONSISTENT_LEARNER",
      desc: "Maintained a 7-day streak!",
      icon: "🔥",
    },
  ];

  return (
    <div className="min-h-screen w-full bg-[#F4EDE4] text-[#2C3E33]">
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b-4 border-border/50 px-4 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <GameButton
              variant="back"
              onClick={() => router.push("/dashboard")}
            >
              <ChevronLeft size={24} strokeWidth={3} />
            </GameButton>
            <h1 className="font-display text-3xl font-black text-primary tracking-tighter text-shadow-sm">
              SignLearn
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <CoinDisplay amount={userCoins} />
            <ProfileDropdown userName={displayName} />
          </div>
        </div>
      </nav>

      <main className="pt-28 pb-12 px-4 max-w-7xl mx-auto space-y-8">
        {/* DYNAMIC HEADER */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 bg-white p-8 rounded-[3rem] border-b-[12px] border-slate-200 shadow-xl relative overflow-hidden">
          <div className="relative z-10 space-y-2">
            <div className="flex items-center gap-2 text-primary font-black uppercase text-xs tracking-widest">
              <ShoppingBag size={16} />
              <span>{tabHeaderInfo[selectedTab].badge}</span>
            </div>
            <h1 className="text-5xl font-black uppercase tracking-tighter">
              {tabHeaderInfo[selectedTab].main}{" "}
              <span className="text-primary">
                {tabHeaderInfo[selectedTab].sub}
              </span>
            </h1>
            <p className="text-muted-foreground font-bold max-w-lg">
              {tabHeaderInfo[selectedTab].desc}
            </p>
          </div>
          <div className="relative z-10">
            <div className="flex p-1 bg-slate-100 rounded-2xl border-2 border-slate-200">
              {["avatars", "badges", "rewards"].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setSelectedTab(tab as any)}
                  className={cn(
                    "px-6 py-2 rounded-xl font-black uppercase text-xs transition-all",
                    selectedTab === tab
                      ? "bg-white text-accent shadow-sm translate-y-[-2px] border-b-4 border-accent"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>
          <Sparkles className="absolute -right-10 -top-10 w-48 h-48 text-blue-50 opacity-50" />
        </div>

        <div className="min-h-[500px]">
          {selectedTab === "avatars" && avatars.length > 0 && (
            <div className="max-w-7xl mx-auto pt-4 px-4 md:px-6">
              {/* 1. DEFINE ANIMATION MAPPING LOGIC */}
              {(() => {
                const getAvatarAnimation = (id: number) => {
                  const map: Record<number, string> = {
                    16: "BreathingIdle", // Avatar 0
                    17: "Cheering",
                    18: "ThoughtfulHeadNod",
                    19: "Salute",
                    20: "Idle",
                  };
                  return map[id];
                };

                const currentAvatar = avatars[currentIndex];
                const animationForUnit = getAvatarAnimation(currentAvatar.id);

                return (
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
                    {/* --- 1. THE COMMAND CENTER (Left on Desktop) --- */}
                    <div className="lg:col-span-5 flex flex-col gap-6 order-2 lg:order-1">
                      {/* UNIT SELECTION RACK */}
                      <div className="bg-white rounded-[2rem] p-5 border-b-[10px] border-slate-200 shadow-xl space-y-4">
                        <div className="flex items-center justify-between px-2">
                          <h3 className="font-black uppercase text-[10px] tracking-widest text-primary">
                            Unit Selection Rack
                          </h3>
                          <div className="bg-slate-100 px-2 py-0.5 rounded-full">
                            <span className="text-[9px] font-black text-slate-500 uppercase">
                              {currentIndex + 1} / {avatars.length}
                            </span>
                          </div>
                        </div>

                        <div className="grid grid-cols-5 gap-2">
                          {avatars.map((skin, index) => {
                            const isSelected = currentIndex === index;
                            const isEquipped =
                              Number(dashboard?.equipped_avatar_id) === skin.id;
                            return (
                              <button
                                key={skin.id}
                                onClick={() => setCurrentIndex(index)}
                                className={cn(
                                  "relative aspect-square rounded-xl overflow-hidden transition-all border-b-[3px] active:border-b-0 active:translate-y-1",
                                  isSelected
                                    ? "bg-primary border-green-800 scale-95 shadow-[inset_0_0_12px_rgba(0,0,0,0.2)]"
                                    : "bg-slate-50 border-slate-200 hover:border-primary/40",
                                )}
                              >
                                <img
                                  src={avatarService.getThumbnailUrl(
                                    skin.folder_name,
                                  )}
                                  className={cn(
                                    "w-full h-full object-cover transition-all duration-300",
                                    !skin.is_owned && "grayscale opacity-40",
                                    isSelected && "scale-110",
                                  )}
                                  alt={skin.name}
                                />
                                {isSelected && (
                                  <div className="absolute inset-0 bg-primary/20" />
                                )}
                                {isEquipped && (
                                  <div className="absolute top-1 left-1 bg-white rounded-full p-0.5 shadow-md">
                                    <CheckCircle2
                                      size={8}
                                      className="text-primary"
                                    />
                                  </div>
                                )}
                              </button>
                            );
                          })}
                        </div>
                      </div>

                      {/* CHARACTER SPECIFICATION CARD */}
                      <div className="bg-white rounded-[2rem] p-10 border-b-[16px] border-slate-200 shadow-2xl flex-1 flex flex-col justify-between">
                        <div className="space-y-6">
                          <div className="flex justify-between items-start gap-4">
                            <div className="space-y-1">
                              <h2 className="text-4xl md:text-5xl font-black uppercase tracking-tighter text-foreground leading-none">
                                {currentAvatar.name}
                              </h2>
                              <div className="flex gap-2">
                                <span
                                  className={cn(
                                    "text-[9px] font-black uppercase tracking-widest px-2 py-0.5 rounded-md",
                                    currentAvatar.price === 0
                                      ? "bg-slate-100 text-slate-500"
                                      : "bg-yellow-100 text-yellow-600",
                                  )}
                                >
                                  {currentAvatar.price === 0
                                    ? "Starter Unit"
                                    : "Legendary Model"}
                                </span>
                              </div>
                            </div>
                            {!currentAvatar.is_owned && (
                              <div className="bg-yellow-400 text-yellow-900 px-4 py-2 rounded-2xl border-b-4 border-yellow-600 font-black text-xl shadow-lg">
                                {currentAvatar.price} 💰
                              </div>
                            )}
                          </div>

                          <p className="text-muted-foreground font-bold text-xs leading-relaxed italic border-l-4 border-primary/20 pl-4">
                            "
                            {currentAvatar.description ||
                              `Specialized training unit programmed for ${animationForUnit} protocols.`}
                            "
                          </p>

                          <div className="space-y-4 pt-2">
                            <p className="text-[10px] font-black uppercase text-primary tracking-[0.2em]">
                              Calibration Data
                            </p>
                            <div className="grid grid-cols-2 gap-6">
                              <StatBar
                                label="Hand Clarity"
                                value={currentIndex % 2 === 0 ? 92 : 85}
                                color="bg-green-500"
                              />
                              <StatBar
                                label="Sync Speed"
                                value={currentIndex % 2 === 0 ? 78 : 94}
                                color="bg-blue-500"
                              />
                            </div>
                          </div>
                        </div>

                        <div className="pt-8">
                          {Number(dashboard?.equipped_avatar_id) ===
                          currentAvatar.id ? (
                            <div className="w-full bg-slate-50 py-5 rounded-3xl text-center border-2 border-dashed border-slate-200">
                              <span className="text-slate-400 font-black uppercase tracking-[0.3em] text-[10px]">
                                Fully Deployed
                              </span>
                            </div>
                          ) : (
                            <GameButton
                              variant={
                                currentAvatar.is_owned ? "duolingo" : "retro"
                              }
                              className="w-full py-2 text-2xl"
                              onClick={() => handleAvatarAction(currentAvatar)}
                              isLoading={isProcessing}
                              disabled={
                                !currentAvatar.is_owned &&
                                userCoins < currentAvatar.price
                              }
                            >
                              {currentAvatar.is_owned ? "ACTIVATE" : "RECRUIT"}
                            </GameButton>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* --- 2. THE HERO SHOWCASE (Right on Desktop) --- */}
                    <div className="lg:col-span-7 relative group order-1 lg:order-2">
                      <div className="h-[450px] md:h-[550px] lg:h-[700px] w-full rounded-[4rem] bg-slate-950 border-4 border-white shadow-2xl overflow-hidden relative">
                        {/* HOLOGRAPHIC OVERLAYS */}
                        <div className="absolute inset-0 pointer-events-none z-10 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.5)_100%)]" />
                        <div className="absolute inset-0 opacity-5 pointer-events-none z-10 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%),linear-gradient(90deg,rgba(255,255,255,0.05),rgba(255,255,255,0),rgba(255,255,255,0.05))] bg-[length:100%_4px,20px_100%]" />

                        <div className="absolute top-10 left-10 z-20 flex items-center gap-3">
                          <div className="w-2 h-2 bg-primary rounded-full animate-pulse shadow-[0_0_10px_#5F7A61]" />
                          <span className="text-white font-black text-[10px] uppercase tracking-[0.4em] opacity-80">
                            {animationForUnit}
                          </span>
                        </div>

                        <div className="absolute top-10 right-10 z-20 text-right bg-black/20 backdrop-blur-md p-4 rounded-3xl border border-white/5">
                          <p className="text-white/40 font-black text-[8px] uppercase tracking-widest leading-none mb-1">
                            Unit ID
                          </p>
                          <p className="text-white font-black text-xl italic tracking-widest uppercase leading-none">
                            #{currentAvatar.id}
                          </p>
                        </div>

                        {/* DYNAMIC ANIMATION TRIGGERED HERE */}
                        <Avatar3DViewer
                          key={`hero-${currentAvatar.id}`}
                          avatarFolder={currentAvatar.folder_name}
                          animationName={animationForUnit}
                          cameraPosition={[1, 0.5, 7.5]}
                          stagePosition={[0, -2.8, 0]}
                        />

                        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 opacity-0 group-hover:opacity-40 transition-all text-center">
                          <p className="text-white font-black text-[8px] uppercase tracking-[0.6em]">
                            Inspect Hardware
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          {selectedTab === "badges" && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {backendBadges.map((badge) => {
                const isOwned = ownedBadges.includes(badge.code);
                return (
                  <div
                    key={badge.code}
                    className={cn(
                      "bg-white rounded-[2.5rem] p-8 border-b-8 border-slate-200 shadow-xl flex flex-col items-center text-center gap-4 transition-all",
                      !isOwned && "grayscale opacity-70",
                    )}
                  >
                    <div
                      className={cn(
                        "w-24 h-24 rounded-full flex items-center justify-center text-5xl shadow-inner border-4",
                        isOwned
                          ? "bg-yellow-50 border-yellow-200"
                          : "bg-slate-100 border-slate-200",
                      )}
                    >
                      {badge.icon}
                    </div>
                    <div>
                      <h3 className="font-black uppercase tracking-tight text-xl">
                        {badge.name}
                      </h3>
                      <p className="text-sm text-muted-foreground font-medium px-4">
                        {badge.desc}
                      </p>
                    </div>
                    <div
                      className={cn(
                        "px-4 py-1 rounded-full font-black text-[10px] uppercase tracking-widest",
                        isOwned
                          ? "bg-primary/10 text-primary"
                          : "bg-slate-100 text-muted-foreground",
                      )}
                    >
                      {isOwned ? "Unlocked" : "Locked"}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {selectedTab === "rewards" && (
            <div className="max-w-5xl mx-auto pt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 bg-white rounded-[3rem] p-8 md:p-12 border-b-[12px] border-slate-200 shadow-2xl items-center">
                <div className="space-y-8 text-center md:text-left order-2 md:order-1">
                  <div className="space-y-4">
                    <div className="inline-flex items-center gap-2 text-yellow-600 font-black uppercase text-xs tracking-widest bg-yellow-50 px-4 py-2 rounded-2xl border-2 border-yellow-100">
                      <Gift size={18} />
                      <span>Daily Loot Box</span>
                    </div>
                    <h2 className="text-6xl font-black uppercase tracking-tighter leading-none">
                      Mystic <br />
                      <span className="text-yellow-500">Chest</span>
                    </h2>
                    <p className="text-muted-foreground font-bold text-lg leading-relaxed">
                      {chestOpen
                        ? "You've claimed today's treasure!"
                        : "Click the chest to reveal what's inside!"}
                    </p>
                  </div>
                  <div
                    className={cn(
                      "p-6 rounded-3xl border-2 flex items-center justify-center md:justify-start gap-4",
                      chestOpen
                        ? "bg-green-50 border-green-200"
                        : "bg-slate-50 border-slate-100",
                    )}
                  >
                    <div
                      className={cn(
                        "w-12 h-12 rounded-full flex items-center justify-center shadow-lg text-2xl",
                        chestOpen ? "bg-green-500" : "bg-yellow-400",
                      )}
                    >
                      {chestOpen ? (
                        <CheckCircle2 className="text-white" />
                      ) : (
                        "💰"
                      )}
                    </div>
                    <div>
                      <p className="text-2xl font-black text-foreground">
                        {chestOpen ? "Claimed" : "100 Coins"}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="relative aspect-square w-full rounded-[2.5rem] bg-slate-950 border-4 border-slate-100 shadow-inner overflow-hidden order-1 md:order-2 group">
                  <div className="absolute inset-0 z-20">
                    <GameChest3D isOpen={chestOpen} onOpen={handleClaimDaily} />
                  </div>
                  {!chestOpen && (
                    <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-30 pointer-events-none flex flex-col items-center gap-1">
                      <ChevronUp
                        size={32}
                        className="text-white animate-bounce"
                      />
                      <div className="bg-white/20 backdrop-blur-md px-6 py-2 rounded-full border-2 border-white/30">
                        <p className="text-white font-black text-xs uppercase tracking-widest shadow-sm">
                          Click to Open
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* EARN COINS GUIDE */}
        <section className="bg-white rounded-[3rem] p-10 border-b-[12px] border-slate-200 shadow-2xl space-y-8 relative overflow-hidden">
          <div className="flex items-center gap-3 relative z-10">
            <div className="bg-yellow-400 p-2 rounded-xl border-b-4 border-yellow-600">
              <Info className="text-white" size={24} />
            </div>
            <h3 className="font-display text-2xl font-black uppercase tracking-tighter">
              Master the Economy
            </h3>
          </div>
          <div className="grid md:grid-cols-3 gap-6 relative z-10">
            <div className="bg-slate-50 p-6 rounded-[2rem] border-b-4 border-slate-200 transition-all hover:scale-[1.02] group">
              <div className="bg-primary/20 w-14 h-14 rounded-2xl flex items-center justify-center mb-6 border-b-4 border-primary/30 group-hover:rotate-6 transition-transform">
                <BookOpen className="text-primary" size={28} />
              </div>
              <h4 className="font-black uppercase text-sm tracking-tight mb-2">
                Knowledge Quests
              </h4>
              <p className="text-sm text-muted-foreground font-medium">
                Earn{" "}
                <span className="text-yellow-600 font-bold">50-100 coins</span>{" "}
                per lesson.
              </p>
            </div>
            <div className="bg-slate-50 p-6 rounded-[2rem] border-b-4 border-slate-200 transition-all hover:scale-[1.02] group">
              <div className="bg-orange-100 w-14 h-14 rounded-2xl flex items-center justify-center mb-6 border-b-4 border-orange-200 group-hover:animate-bounce">
                <Flame className="text-orange-500 fill-orange-500" size={28} />
              </div>
              <h4 className="font-black uppercase text-sm tracking-tight mb-2">
                Daily Momentum
              </h4>
              <p className="text-sm text-muted-foreground font-medium">
                Maintain your streak for multipliers.
              </p>
            </div>
            <div className="bg-slate-50 p-6 rounded-[2rem] border-b-4 border-slate-200 transition-all hover:scale-[1.02] group">
              <div className="bg-yellow-100 w-14 h-14 rounded-2xl flex items-center justify-center mb-6 border-b-4 border-yellow-200 group-hover:-rotate-6 transition-transform">
                <Star className="text-yellow-600 fill-yellow-600" size={28} />
              </div>
              <h4 className="font-black uppercase text-sm tracking-tight mb-2">
                Epic Milestones
              </h4>
              <p className="text-sm text-muted-foreground font-medium">
                Unlock badges to claim massive treasure.
              </p>
            </div>
          </div>
          <HelpCircle className="absolute -bottom-10 -right-10 w-64 h-64 text-slate-100/50" />
        </section>

        {/* START TRAINING SECTION */}
        <div className="bg-primary p-8 rounded-[3rem] text-white border-b-8 border-[#4a5f4b] shadow-xl flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="flex items-center gap-6">
            <div className="bg-white/20 p-4 rounded-3xl backdrop-blur-md">
              <Zap size={40} className="text-yellow-300 fill-yellow-300" />
            </div>
            <div>
              <h3 className="text-2xl font-black uppercase tracking-tight">
                Need more coins?
              </h3>
              <p className="font-bold opacity-80">
                Complete daily quests and maintain your streak to fill your
                coffers!
              </p>
            </div>
          </div>
          <Link href="/lessons">
            <GameButton variant="retro" className="px-10 py-6">
              Start Training
            </GameButton>
          </Link>
        </div>
      </main>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background gap-4">
      <div className="w-16 h-16 border-8 border-primary/20 border-t-primary rounded-full animate-spin" />
      <p className="font-black uppercase tracking-widest text-primary animate-pulse">
        Syncing Inventory...
      </p>
    </div>
  );
}

export default function ShopPage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <ShopContent />
    </Suspense>
  );
}
