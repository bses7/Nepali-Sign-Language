"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { GameButton } from "@/components/game-button";
import { CoinDisplay } from "@/components/game-stats";
import { ProfileDropdown } from "@/components/profile-dropdown";
import { Avatar3DViewer } from "@/components/avatar-viewer-3d";
import { useAuthStore } from "@/lib/store/auth";
import { usersService } from "@/lib/api/users";
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
  ChevronUp, // Added for the chest indicator
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

  const [ownedBadges, setOwnedBadges] = useState<string[]>([]);
  const [isClaiming, setIsClaiming] = useState(false);
  const [chestOpen, setChestOpen] = useState(false);

  // Dynamic Header Content Mapping
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
    if (dashboard) {
      setChestOpen(!dashboard.can_claim_daily);
    }
  }, [dashboard]);

  useEffect(() => {
    const tab = searchParams.get("tab") as any;
    if (tab === "avatars" || tab === "badges" || tab === "rewards") {
      setSelectedTab(tab);
    }
  }, [searchParams]);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }
    fetchDashboard();
    usersService.getUserBadges().then((res) => {
      if (res.success)
        setOwnedBadges(res.data?.map((b: any) => b.badge_code) || []);
    });
  }, [isAuthenticated, router, fetchDashboard]);

  const handleClaimDaily = async () => {
    if (chestOpen || isClaiming || !dashboard?.can_claim_daily) return;

    setIsClaiming(true);
    try {
      const res = await usersService.claimDailyReward();
      if (res.success) {
        setChestOpen(true);
        toast.success("Successfully claimed 100 coins!");
        fetchDashboard();
      } else {
        toast.error(res.error || "Chest is empty! Come back tomorrow.");
      }
    } catch (err) {
      toast.error("Network error. Try again later.");
    } finally {
      setIsClaiming(false);
    }
  };

  if (isDashboardLoading) return <LoadingScreen />;

  const userCoins = dashboard?.coins ?? 0;
  const displayName = dashboard?.first_name || user?.first_name || "Learner";

  const avatarSkins = [
    {
      id: "default",
      name: "Classic Sage",
      price: 0,
      folder: "default",
      owned: true,
    },
    {
      id: "indigo",
      name: "Deep Indigo",
      price: 500,
      folder: "indigo",
      owned: true,
    },
    {
      id: "gold",
      name: "Golden King",
      price: 2000,
      folder: "gold",
      owned: false,
    },
    {
      id: "emerald",
      name: "Forest Ranger",
      price: 1200,
      folder: "emerald",
      owned: false,
    },
  ];

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
        {/* DYNAMIC SHOP HEADER */}
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
          {selectedTab === "avatars" && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {avatarSkins.map((skin) => (
                <div
                  key={skin.id}
                  className="group bg-white rounded-[2.5rem] p-4 border-b-8 border-slate-200 shadow-xl hover:-translate-y-1 transition-all"
                >
                  <div className="aspect-[4/5] w-full rounded-[2rem] bg-slate-900 mb-4 overflow-hidden relative border-4 border-slate-100 shadow-inner">
                    <Avatar3DViewer
                      avatarFolder={skin.folder}
                      className="scale-[1.25] translate-y-4"
                    />
                    {!skin.owned && (
                      <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-md px-3 py-1 rounded-xl border border-white/20">
                        <span className="text-white font-black text-[10px] uppercase tracking-widest">
                          Locked
                        </span>
                      </div>
                    )}
                  </div>
                  <div className="px-2 space-y-4 text-center">
                    <h3 className="font-black uppercase tracking-tight text-lg">
                      {skin.name}
                    </h3>
                    <GameButton
                      variant={skin.owned ? "duolingo" : "jelly"}
                      className="w-full"
                      disabled={!skin.owned && userCoins < skin.price}
                    >
                      {skin.owned ? "EQUIP" : `${skin.price} 💰`}
                    </GameButton>
                  </div>
                </div>
              ))}
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
                        ? "You've claimed today's treasure! Check back in 24 hours."
                        : "Click the chest to reveal what's inside!"}
                    </p>
                  </div>
                  <div
                    className={cn(
                      "p-6 rounded-3xl border-2 flex items-center justify-center md:justify-start gap-4 transition-all",
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
                      <p className="text-[10px] font-black uppercase text-muted-foreground">
                        Daily Login Bonus
                      </p>
                    </div>
                  </div>
                </div>

                {/* INTERACTIVE 3D CHEST WITH UPARROW */}
                {/* 2. INTERACTIVE 3D CHEST (Inside your rewards tab) */}
                <div className="relative aspect-square w-full rounded-[2.5rem] bg-slate-950 border-4 border-slate-100 shadow-inner overflow-hidden order-1 md:order-2 group">
                  <div className="absolute inset-0 z-20">
                    <GameChest3D isOpen={chestOpen} onOpen={handleClaimDaily} />
                  </div>

                  {/* Background Glow Overlay (Keep this very subtle) */}
                  <div
                    className={cn(
                      "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-80 h-80 rounded-full blur-[100px] transition-all duration-1000 pointer-events-none",
                      chestOpen ? "bg-yellow-400/10" : "bg-blue-500/5",
                    )}
                  />

                  {/* Floating Indicator */}
                  {!chestOpen && (
                    <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-30 pointer-events-none flex flex-col items-center gap-1">
                      <ChevronUp
                        size={32}
                        className="text-white animate-bounce"
                      />
                      <div className="bg-white/20 backdrop-blur-md px-6 py-2 rounded-full border-2 border-white/30 animate-pulse">
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

        {/* HOW TO EARN COINS GUIDE */}
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
            <div className="bg-slate-50 p-6 rounded-[2rem] border-b-4 border-slate-200 transition-all hover:scale-[1.02] hover:border-primary/40 group">
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
            <div className="bg-slate-50 p-6 rounded-[2rem] border-b-4 border-slate-200 transition-all hover:scale-[1.02] hover:border-orange-400 group">
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
            <div className="bg-slate-50 p-6 rounded-[2rem] border-b-4 border-slate-200 transition-all hover:scale-[1.02] hover:border-yellow-400 group">
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
          <Link href="/dashboard">
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
