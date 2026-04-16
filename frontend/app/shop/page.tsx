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
  const [isClaiming, setIsClaiming] = useState(false);
  const [chestOpen, setChestOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [localShopError, setLocalShopError] = useState<string | null>(null);
  const [badgesList, setBadgesList] = useState<any[]>([]);

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
      desc: "Your legendary achievements and certificates of mastery.",
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

    usersService.getAllBadges().then((res) => {
      if (res.success) setBadgesList(res.data || []);
    });
  }, [isAuthenticated, router, fetchDashboard]);

  const [showCelebration, setShowCelebration] = useState(false);
  const [activeAnimation, setActiveAnimation] = useState("Idle");
  const [isShopMode, setIsShopMode] = useState(false);

  const currentAvatar = avatars[currentIndex];

  const playRandomShopAnimation = () => {
    const availableAnims = currentAvatar?.attributes?.shop_animations || [];

    if (availableAnims.length > 0) {
      const randomIndex = Math.floor(Math.random() * availableAnims.length);
      setActiveAnimation(availableAnims[randomIndex]);
      setIsShopMode(true);
    } else {
      // Fallback if this avatar has no shop folder animations
      setActiveAnimation("Idle");
      setIsShopMode(false);
    }
  };

  useEffect(() => {
    if (currentAvatar) {
      playRandomShopAnimation();
      setShowCelebration(false);
    }
  }, [currentIndex, avatars]);

  const handleAvatarAction = async (avatar: AvatarItem) => {
    if (isProcessing) return;
    setLocalShopError(null);
    setIsProcessing(true);

    try {
      if (avatar.is_owned) {
        const res = await avatarService.equipAvatar(avatar.id);
        if (res.success) {
          toast.success(`${avatar.name} Equipped!`);
          await fetchDashboard();
        }
      } else {
        if (userCoins < avatar.price) {
          setLocalShopError("Insufficient coins! Go finish some lessons.");
          toast.error("Insufficient coins");
          setIsProcessing(false);
          return;
        }

        const res = await avatarService.buyAvatar(avatar.id);

        if (res.success) {
          playRandomShopAnimation();
          setShowCelebration(true);

          const storeRes = await avatarService.getAvatarStore();
          if (storeRes.success) setAvatars(storeRes.data || []);
          await fetchDashboard();

          setTimeout(() => {
            setShowCelebration(false);
            setActiveAnimation("Idle");
            setIsShopMode(false);
          }, 4000);
        } else {
          const errorMsg = res.error || "Transaction failed.";
          setLocalShopError(errorMsg);
          toast.error(errorMsg);
        }
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClaimDaily = async () => {
    console.log("Claim Daily Attempted");
    console.log("Current Chest State:", chestOpen);
    console.log("Can Claim State:", dashboard?.can_claim_daily);

    if (chestOpen || isClaiming) {
      console.log("Blocked: Chest already open or currently claiming");
      return;
    }

    if (!dashboard?.can_claim_daily) {
      console.log("Blocked: Dashboard says user cannot claim");
      toast.error("Already claimed today!");
      setChestOpen(true);
      return;
    }

    setIsClaiming(true);
    try {
      const res = await usersService.claimDailyReward();
      console.log("API Response:", res);

      if (res.success) {
        setChestOpen(true);
        toast.success("Congratulations! You’ve claimed 100 coins.", {
          icon: "💰",
        });
        await fetchDashboard();
      } else {
        toast.error(res.error || "Failed to claim reward");
      }
    } catch (error) {
      console.error("Claim error:", error);
      toast.error("Connection error to the Vault.");
    } finally {
      setIsClaiming(false);
    }
  };

  if (isDashboardLoading && !dashboard) return <LoadingScreen />;

  const userCoins = dashboard?.coins ?? 0;
  const displayName = dashboard?.first_name || user?.first_name || "Learner";

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
              {(() => {
                const currentAvatar = avatars[currentIndex];

                const activeAnim = showCelebration ? "Salute" : "Idle";

                return (
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
                    {/* --- 1. LEFT SIDE: COMMAND CENTER (Roster & Stats) --- */}
                    <div className="lg:col-span-5 flex flex-col gap-6 order-2 lg:order-1">
                      {/* UNIT SELECTION RACK */}
                      <div className="bg-white rounded-[2rem] p-5 border-b-[10px] border-slate-200 shadow-xl space-y-4">
                        <div className="flex items-center justify-between px-2">
                          <h3 className="font-black uppercase text-[10px] tracking-widest text-primary">
                            Characters
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
                                onClick={() => {
                                  setCurrentIndex(index);
                                  setShowCelebration(false);
                                  setLocalShopError(null);
                                }}
                                className={cn(
                                  "relative aspect-square rounded-xl overflow-hidden transition-all border-b-[3px] active:border-b-0 active:translate-y-1",
                                  isSelected
                                    ? "bg-primary border-green-800 scale-95 shadow-inner"
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
                                  <div className="absolute inset-0 bg-primary/10" />
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
                      {/* CHARACTER SPECIFICATION CARD */}
                      <div className="bg-white rounded-[2.5rem] p-6 border-b-[12px] border-slate-200 shadow-2xl flex-1 flex flex-col justify-between relative overflow-hidden">
                        {/* Blueprint Background Effect */}
                        <div
                          className="absolute inset-0 opacity-[0.03] pointer-events-none"
                          style={{
                            backgroundImage: `radial-gradient(#2C3E33 1px, transparent 1px)`,
                            backgroundSize: "20px 20px",
                          }}
                        />

                        <div className="space-y-5 relative z-10">
                          {/* 1. IDENTITY HEADER */}
                          <div className="flex justify-between items-start">
                            <div className="space-y-1">
                              <div className="flex items-center gap-2">
                                {(() => {
                                  const type =
                                    currentAvatar.attributes?.type?.toLowerCase() ||
                                    "common";
                                  const rarityConfigs = {
                                    common:
                                      "bg-gray-50 text-gray-500 border-gray-100",
                                    rare: "bg-orange-50 text-orange-600 border-orange-200",
                                    legendary:
                                      "bg-purple-50 text-purple-600 border-purple-200",
                                  };
                                  const config =
                                    rarityConfigs[
                                      type as keyof typeof rarityConfigs
                                    ] || rarityConfigs.common;

                                  return (
                                    <span
                                      className={cn(
                                        "text-[9px] font-black uppercase tracking-[0.2em] px-2 py-0.5 rounded-full border shadow-sm",
                                        config,
                                      )}
                                    >
                                      {type}
                                    </span>
                                  );
                                })()}
                              </div>
                              <h2 className="text-3xl font-black uppercase tracking-tighter text-slate-800 leading-none">
                                {currentAvatar.name}
                              </h2>
                            </div>

                            {!currentAvatar.is_owned && (
                              <div className="bg-yellow-400 text-yellow-900 px-3 py-1 rounded-xl border-b-4 border-yellow-600 font-black text-lg shadow-md transform rotate-2">
                                {currentAvatar.price} 💰
                              </div>
                            )}
                          </div>

                          {/* 2. VISUAL DNA (Colors) */}
                          <div className="grid grid-cols-2 gap-2">
                            <AttributeChip
                              label="Skin"
                              color={currentAvatar.attributes?.skin_color}
                              value={currentAvatar.attributes?.skin_color}
                            />
                            <AttributeChip
                              label="Hair"
                              color={currentAvatar.attributes?.hair_color}
                              value={currentAvatar.attributes?.hair_color}
                            />
                            <AttributeChip
                              label="Eyes"
                              color={currentAvatar.attributes?.eye_color}
                              value={currentAvatar.attributes?.eye_color}
                            />
                            <AttributeChip
                              label="Outfit"
                              color={currentAvatar.attributes?.clothing_color}
                              value={currentAvatar.attributes?.clothing_color}
                            />
                          </div>

                          {/* 3. PHYSICAL SPECIFICATIONS (Face + Hardware) */}
                          <div className="bg-slate-50 rounded-2xl border-b-2 border-slate-100 overflow-hidden">
                            {/* Face Row */}
                            <div className="p-3 border-b border-white flex items-center gap-3">
                              <div className="w-8 h-8 rounded-lg bg-white flex items-center justify-center border border-slate-200 text-primary shadow-sm">
                                <Sparkles size={16} strokeWidth={2.5} />
                              </div>
                              <div className="flex flex-col">
                                <span className="text-[7px] font-black text-slate-400 uppercase tracking-widest">
                                  Face Shape
                                </span>
                                <span className="text-xs font-bold text-slate-700 capitalize">
                                  {currentAvatar.attributes?.face_shape?.replace(
                                    "_",
                                    " ",
                                  )}
                                </span>
                              </div>
                            </div>

                            {/* Hardware Row */}
                            <div className="p-3 bg-slate-50/50">
                              <span className="text-[7px] font-black text-slate-400 uppercase tracking-widest block mb-2">
                                Hardware Mods
                              </span>
                              <div className="flex flex-wrap gap-1.5">
                                {currentAvatar.attributes?.accessories?.length >
                                0 ? (
                                  currentAvatar.attributes.accessories.map(
                                    (acc: string) => (
                                      <div
                                        key={acc}
                                        className="flex items-center gap-1.5 bg-white px-2 py-1 rounded-lg border border-slate-200 shadow-sm"
                                      >
                                        <div className="w-1 h-1 rounded-full bg-primary animate-pulse" />
                                        <span className="text-[9px] font-black text-slate-600 uppercase tracking-tight">
                                          {acc.replace("_", " ")}
                                        </span>
                                      </div>
                                    ),
                                  )
                                ) : (
                                  <span className="text-[9px] font-bold text-slate-400 italic px-1">
                                    Standard Config
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* ACTION AREA */}
                        <div className="pt-4 relative z-10">
                          {localShopError && (
                            <div className="absolute -top-8 left-0 right-0 animate-wiggle z-20">
                              <div className="bg-destructive text-white px-3 py-1.5 rounded-lg border-b-2 border-red-900 shadow-lg text-center">
                                <span className="text-[9px] font-black uppercase">
                                  ⚠️ {localShopError}
                                </span>
                              </div>
                            </div>
                          )}

                          {Number(dashboard?.equipped_avatar_id) ===
                          currentAvatar.id ? (
                            <div className="w-full bg-gray-50 py-6 rounded-2xl text-center border-2 border-dashed border-gray-200 flex items-center justify-center gap-2">
                              <span className="text-gray-600 font-black uppercase tracking-[0.1em] text-xl">
                                Equipped
                              </span>
                            </div>
                          ) : (
                            <GameButton
                              variant={
                                currentAvatar.is_owned ? "duolingo" : "retro"
                              }
                              className={cn(
                                "w-full py-2 text-xl shadow-lg transition-all active:scale-95",
                                !currentAvatar.is_owned &&
                                  (dashboard?.coins || 0) < currentAvatar.price,
                              )}
                              onClick={() => handleAvatarAction(currentAvatar)}
                              isLoading={isProcessing}
                            >
                              {currentAvatar.is_owned ? "EQUIP" : "PURCHASE"}
                            </GameButton>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="lg:col-span-7 relative group order-1 lg:order-2">
                      <div className="h-[450px] md:h-[550px] lg:h-[700px] w-full rounded-[4rem] bg-slate-950 border-4 border-white shadow-2xl overflow-hidden relative">
                        <div className="absolute inset-0 pointer-events-none z-10 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.5)_100%)]" />
                        <div className="absolute inset-0 opacity-5 pointer-events-none z-10 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%),linear-gradient(90deg,rgba(255,255,255,0.05),rgba(255,255,255,0),rgba(255,255,255,0.05))] bg-[length:100%_4px,20px_100%]" />

                        {/* HUD STATUS */}
                        <div className="absolute top-10 left-10 z-20 flex items-center gap-3">
                          <div
                            className={cn(
                              "w-2 h-2 rounded-full animate-pulse shadow-[0_0_10px]",
                              showCelebration
                                ? "bg-yellow-400 shadow-yellow-400"
                                : "bg-primary shadow-primary",
                            )}
                          />
                          <span className="text-white font-black text-[10px] uppercase tracking-[0.4em] opacity-80">
                            {showCelebration ? "SUCCESS" : "PREVIEW"}
                          </span>
                        </div>

                        <Avatar3DViewer
                          key={`hero-${currentAvatar.id}-${activeAnimation}`}
                          avatarFolder={currentAvatar.folder_name}
                          animationName={activeAnimation}
                          isShopAnimation={isShopMode}
                          cameraPosition={[0, 0.5, 7.5]}
                          stagePosition={[0, -2.5, 0.1]}
                        />

                        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 opacity-0 group-hover:opacity-40 transition-all text-center">
                          <p className="text-white font-black text-[8px] uppercase tracking-[0.6em]">
                            Rotate Model
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
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
              {badgesList.map((badge) => {
                // Logic: Use is_earned directly from backend
                const isEarned = badge.is_earned;
                // Construct full URL for icon
                const iconUrl = badge.icon_url.startsWith("http")
                  ? badge.icon_url
                  : `http://localhost:8000${badge.icon_url}`;

                return (
                  <div
                    key={badge.badge_code}
                    className={cn(
                      "bg-white rounded-[2.5rem] p-8 border-b-8 border-slate-200 shadow-xl flex flex-col items-center text-center gap-5 transition-all duration-500",
                      !isEarned && "grayscale opacity-60",
                    )}
                  >
                    {/* BADGE ICON CONTAINER */}
                    <div className="relative group/badge">
                      <div
                        className={cn(
                          "w-28 h-28 rounded-full flex items-center justify-center p-4 shadow-inner border-4 transition-transform group-hover/badge:scale-110",
                          isEarned
                            ? "bg-yellow-50 border-yellow-200"
                            : "bg-slate-100 border-slate-200",
                        )}
                      >
                        <img
                          src={iconUrl}
                          alt={badge.name}
                          className="w-full h-full object-contain drop-shadow-md"
                          onError={(e) => {
                            // Fallback to a placeholder if image fails to load
                            (e.target as HTMLImageElement).src =
                              "https://ui-avatars.com/api/?name=" + badge.name;
                          }}
                        />
                      </div>

                      {/* Sparkle effect for earned badges */}
                      {isEarned && (
                        <Sparkles
                          className="absolute -top-2 -right-2 text-yellow-400 animate-pulse"
                          size={24}
                        />
                      )}
                    </div>

                    {/* BADGE INFO */}
                    <div className="space-y-2">
                      <h3 className="font-black uppercase tracking-tight text-xl text-foreground">
                        {badge.name}
                      </h3>
                      <p className="text-xs text-muted-foreground font-bold leading-relaxed px-4">
                        {badge.description}
                      </p>
                    </div>

                    {/* STATUS FOOTER */}
                    <div className="w-full pt-2 border-t border-slate-50 flex flex-col items-center gap-2">
                      <div
                        className={cn(
                          "px-6 py-1.5 rounded-full font-black text-[10px] uppercase tracking-[0.2em] shadow-sm",
                          isEarned
                            ? "bg-primary/10 text-primary border border-primary/20"
                            : "bg-slate-100 text-slate-400 border border-slate-200",
                        )}
                      >
                        {isEarned ? "Unlocked" : "Locked"}
                      </div>

                      {isEarned && badge.earned_at && (
                        <p className="text-[9px] font-black text-muted-foreground/50 uppercase tracking-widest">
                          Earned:{" "}
                          {new Date(badge.earned_at).toLocaleDateString()}
                        </p>
                      )}
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

                <div className="relative aspect-square w-full rounded-[2.5rem] bg-slate-950 border-4 border-slate-100 shadow-inner overflow-hidden order-1 md:order-2 group">
                  {/* 1. THE 3D SCENE (Purely visual now) */}
                  <div className="absolute inset-0 z-10 pointer-events-none">
                    <GameChest3D isOpen={chestOpen} onOpen={() => {}} />
                  </div>

                  {/* 2. THE INVISIBLE CLICK OVERLAY (The Fix) */}
                  {!chestOpen && (
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        console.log("Invisible overlay clicked");
                        handleClaimDaily();
                      }}
                      className="absolute inset-0 z-50 w-full h-full bg-transparent cursor-pointer"
                      aria-label="Open Chest"
                    />
                  )}

                  {/* 3. GLOW EFFECTS (Pointer events none) */}
                  <div
                    className={cn(
                      "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-80 h-80 rounded-full blur-[100px] transition-all duration-1000 pointer-events-none z-0",
                      chestOpen ? "bg-yellow-400/20" : "bg-blue-500/10",
                    )}
                  />

                  {/* 4. FLOATING INDICATOR (Pointer events none) */}
                  {!chestOpen && (
                    <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-40 pointer-events-none flex flex-col items-center gap-1">
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

function AttributeChip({
  label,
  value,
  color,
  icon: Icon,
}: {
  label: string;
  value?: string;
  color?: string;
  icon?: any;
}) {
  return (
    <div className="flex items-center gap-3 bg-slate-50 p-3 rounded-xl border-b-2 border-slate-100 hover:bg-slate-100 transition-colors group">
      <div className="relative shrink-0">
        {color ? (
          <div
            className="w-8 h-8 rounded-lg border-2 border-white shadow-sm ring-1 ring-slate-200"
            style={{ backgroundColor: color }}
          />
        ) : (
          <div className="w-8 h-8 rounded-lg bg-white flex items-center justify-center border-2 border-slate-100 text-primary">
            {Icon && <Icon size={14} strokeWidth={2.5} />}
          </div>
        )}
      </div>
      <div className="flex flex-col min-w-0">
        <span className="text-[7px] font-black text-slate-400 uppercase tracking-widest leading-none mb-0.5">
          {label}
        </span>
        <span className="text-[10px] font-bold text-slate-700 capitalize truncate">
          {value || color || "N/A"}
        </span>
      </div>
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
