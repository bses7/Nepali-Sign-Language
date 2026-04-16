"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { usersService } from "@/lib/api/users";
import { lessonsService } from "@/lib/api/lessons"; // Added to fetch signs for math
import { GameButton } from "@/components/game-button";
import { Avatar3DViewer } from "@/components/avatar-viewer-3d";
import { CoinDisplay } from "@/components/game-stats";
import { ProfileDropdown } from "@/components/profile-dropdown";
import { FcGoogle } from "react-icons/fc"; // Better Google Icon
import { FaGithub } from "react-icons/fa"; // Better GitHub Icon
import {
  Mail,
  Phone,
  ShieldCheck,
  Calendar,
  Trophy,
  BadgeCheck,
  ChevronLeft,
  Sparkles,
  Fingerprint,
  Link as LinkIcon,
  CheckCircle2,
  Zap,
  Target,
  BookOpen,
  GraduationCap,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import Link from "next/link";

export default function ProfilePage() {
  const router = useRouter();
  const { isAuthenticated, dashboard, fetchDashboard, user } = useAuthStore();
  const [badges, setBadges] = useState<any[]>([]);
  const [signs, setSigns] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }

    const loadProfileData = async () => {
      setIsLoading(true);
      await fetchDashboard();
      const [badgeRes, signsRes] = await Promise.all([
        usersService.getAllBadges(),
        lessonsService.getSigns(),
      ]);
      if (badgeRes.success) setBadges(badgeRes.data || []);
      if (signsRes.success) setSigns(signsRes.data || []);
      setIsLoading(false);
    };

    loadProfileData();
  }, [isAuthenticated, router, fetchDashboard]);

  const handleLinkAccount = (provider: "google" | "github") => {
    toast.info(`Syncing with ${provider}...`);
    window.location.href = `http://localhost:8000/api/v1/auth/login/${provider}`;
  };

  if (isLoading || !dashboard) return <LoadingScreen />;

  // --- MATH FOR MASTERY SECTION ---
  const vTotal = signs.filter((s) => s.category === "vowel").length || 13;
  const vDone = signs.filter(
    (s) => s.category === "vowel" && s.is_completed,
  ).length;
  const cTotal = signs.filter((s) => s.category === "consonant").length || 36;
  const cDone = signs.filter(
    (s) => s.category === "consonant" && s.is_completed,
  ).length;

  const earnedBadges = badges.filter((b) => b.is_earned);
  const displayName = dashboard.first_name + " " + dashboard.last_name;

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
            <h1 className="font-display text-3xl font-black text-primary tracking-tighter">
              Profile
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <CoinDisplay amount={dashboard.coins} />
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

      <main className="pt-28 pb-12 px-4 max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* LEFT COLUMN: IDENTITY & AUTH */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-white rounded-[3rem] p-6 border-b-[12px] border-slate-200 shadow-2xl space-y-4">
            <div className="relative h-[500px] w-full rounded-[2.5rem] overflow-hidden bg-slate-900 border-4 border-slate-100 shadow-inner">
              <Avatar3DViewer
                avatarFolder={dashboard.equipped_avatar_folder}
                animationName="Idle"
              />
              <div className="absolute top-6 left-6 bg-primary text-white px-4 py-1 rounded-full font-black text-xs uppercase tracking-widest shadow-lg">
                Level {dashboard.level}
              </div>
            </div>
            <div className="text-center">
              <h2 className="text-3xl font-black uppercase tracking-tighter">
                {displayName}
              </h2>
              <p className="text-primary font-black uppercase text-[10px] tracking-[0.2em]">
                {dashboard.role} Unit
              </p>
            </div>
          </div>

          <div className="bg-white rounded-[2.5rem] p-8 border-b-8 border-slate-200 shadow-xl space-y-6">
            <h3 className="font-black uppercase text-xs tracking-widest text-muted-foreground flex items-center gap-2">
              <ShieldCheck size={16} /> Authentication Security
            </h3>
            <div className="space-y-4">
              <SocialLinkItem
                icon={<FcGoogle size={24} />}
                label="Google Sync"
                isLinked={!!dashboard.google_id}
                onLink={() => handleLinkAccount("google")}
              />
              <SocialLinkItem
                icon={<FaGithub size={24} className="text-slate-900" />}
                label="GitHub Sync"
                isLinked={!!dashboard.github_id}
                onLink={() => handleLinkAccount("github")}
              />
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: DATA & ANALYTICS */}
        <div className="lg:col-span-7 space-y-6">
          {/* PLAYER DOSSIER */}
          <div className="bg-white rounded-[3rem] p-8 border-b-[12px] border-slate-200 shadow-2xl space-y-8">
            <div className="flex items-center gap-3">
              <div className="bg-primary/10 p-3 rounded-2xl">
                <Fingerprint className="text-primary" />
              </div>
              <h3 className="text-2xl font-black uppercase tracking-tighter">
                User Information
              </h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <InfoItem
                icon={<Mail size={18} />}
                label="Mail"
                value={user?.email || "N/A"}
              />
              <InfoItem
                icon={<Phone size={18} />}
                label="Phone Number"
                value={dashboard.phone_number || "Disconnected"}
              />
              <InfoItem
                icon={<Calendar size={18} />}
                label="Joined"
                value="March 2026"
              />
              <InfoItem
                icon={<BadgeCheck size={18} />}
                label="Mastery"
                value={`${dashboard.progress_percentage}%`}
              />
            </div>
          </div>

          {/* HALL OF FAME (Badges) */}
          <div className="bg-white rounded-[3rem] p-8 border-b-[12px] border-slate-200 shadow-2xl space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-2xl font-black uppercase tracking-tighter flex items-center gap-2">
                <Trophy className="text-warning fill-warning/20" /> Hall of Fame
              </h3>
              <Link href="/shop?tab=badges">
                <span className="text-xs font-black text-primary uppercase hover:underline cursor-pointer">
                  Explore Vault
                </span>
              </Link>
            </div>
            <div className="grid grid-cols-4 sm:grid-cols-6 gap-3">
              {earnedBadges.map((badge) => (
                <div
                  key={badge.id}
                  className="group relative aspect-square bg-slate-50 rounded-2xl border-2 border-slate-100 flex items-center justify-center p-2 hover:scale-110 transition-all cursor-help shadow-sm"
                  title={badge.name}
                >
                  <img
                    src={`http://localhost:8000${badge.icon_url}`}
                    alt={badge.name}
                    className="w-full h-full object-contain"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* --- NEW SECTION: MASTERY ANALYTICS --- */}
          <div className="bg-white rounded-[3rem] p-8 border-b-[12px] border-slate-200 shadow-2xl space-y-6">
            <h3 className="text-2xl font-black uppercase tracking-tighter flex items-center gap-2">
              <Target className="text-accent" /> Capability Analysis
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <MasteryProgress
                label="Vowels"
                done={vDone}
                total={vTotal}
                icon={<GraduationCap size={16} />}
                color="bg-primary"
              />
              <MasteryProgress
                label="Consonants"
                done={cDone}
                total={cTotal}
                icon={<BookOpen size={16} />}
                color="bg-secondary"
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// --- REFINED SUB-COMPONENTS ---

function MasteryProgress({ label, done, total, icon, color }: any) {
  const percent = Math.round((done / total) * 100);
  return (
    <div className="bg-slate-50 p-5 rounded-3xl border-b-4 border-slate-100 space-y-3">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2 font-black text-[10px] uppercase text-muted-foreground">
          {icon} <span>{label}</span>
        </div>
        <span className="font-black text-primary text-xs">{percent}%</span>
      </div>
      <div className="h-3 w-full bg-slate-200 rounded-full overflow-hidden shadow-inner">
        <div
          className={cn("h-full transition-all duration-1000", color)}
          style={{ width: `${percent}%` }}
        />
      </div>
      <p className="text-[9px] font-bold text-muted-foreground uppercase text-right tracking-tighter">
        {done} / {total} Signs
      </p>
    </div>
  );
}

function SocialLinkItem({ icon, label, isLinked, onLink }: any) {
  return (
    <div
      className={cn(
        "flex items-center justify-between p-4 rounded-2xl border-2 transition-all",
        isLinked
          ? "bg-primary/5 border-primary/20 shadow-inner"
          : "bg-slate-50 border-slate-200 border-dashed",
      )}
    >
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-white rounded-2xl flex items-center justify-center border-b-4 border-slate-100 shadow-sm">
          {icon}
        </div>
        <div className="flex flex-col">
          <span className="font-black text-sm uppercase tracking-tight leading-none mb-1">
            {label}
          </span>
          <p
            className={cn(
              "text-[9px] font-black uppercase tracking-widest",
              isLinked ? "text-primary" : "text-muted-foreground/50",
            )}
          >
            {isLinked ? "Authenticated" : "Not Connected"}
          </p>
        </div>
      </div>
      {isLinked ? (
        <CheckCircle2 size={24} className="text-primary mr-2" />
      ) : (
        <button
          onClick={onLink}
          className="bg-white hover:bg-primary hover:text-white text-primary px-4 py-2 rounded-xl border-b-4 border-slate-200 hover:border-green-800 transition-all font-black text-[10px] uppercase active:translate-y-1 active:border-b-0"
        >
          Sync Now
        </button>
      )}
    </div>
  );
}

function InfoItem({
  icon,
  label,
  value,
}: {
  icon: any;
  label: string;
  value: string;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2 text-muted-foreground">
        {icon}
        <span className="text-[9px] font-black uppercase tracking-[0.2em]">
          {label}
        </span>
      </div>
      <p className="font-bold text-foreground pl-6 break-all leading-none">
        {value}
      </p>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-[#F4EDE4] gap-4">
      <div className="w-16 h-16 border-8 border-primary/20 border-t-primary rounded-full animate-spin" />
      <p className="font-black uppercase tracking-widest text-primary animate-pulse">
        Scanning Player DNA...
      </p>
    </div>
  );
}
