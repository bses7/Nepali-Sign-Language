"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { lessonsService } from "@/lib/api/lessons";
import { GameButton } from "@/components/game-button";
import { Sign3DViewer } from "@/components/sign-3d-viewer";
import { CoinDisplay } from "@/components/game-stats";
import {
  ArrowLeft,
  Star,
  CheckCircle2,
  Zap,
  ShieldCheck,
  Trophy,
  Video,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

export default function LessonDetailPage() {
  const { id } = useParams();
  const router = useRouter();
  const { dashboard, fetchDashboard } = useAuthStore();

  const [sign, setSign] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isMastered, setIsMastered] = useState(false);
  const [isCompleting, setIsCompleting] = useState(false);

  useEffect(() => {
    const loadData = async () => {
      const detailRes = await lessonsService.getSignById(id as string);
      const listRes = await lessonsService.getSigns();

      if (detailRes.success) {
        setSign(detailRes.data);
        if (listRes.success) {
          const signProgress = listRes.data?.find(
            (s: any) => s.id === Number(id),
          );
          if (signProgress) {
            setIsMastered(signProgress.is_completed);
          }
        }
      } else {
        toast.error("Sign data not found!");
        router.push("/lessons");
      }
      setIsLoading(false);
      fetchDashboard();
    };
    loadData();
  }, [id, fetchDashboard, router]);

  const handleComplete = async () => {
    if (isMastered || isCompleting) return;
    setIsCompleting(true);
    const res = await lessonsService.completeSign(Number(id));
    if (res.success) {
      setIsMastered(true);
      toast.success("LEVEL MASTERED!", {
        description: "+50 XP and 100 Coins earned!",
        icon: <Trophy className="text-yellow-500" />,
      });
      fetchDashboard();
    } else {
      toast.error(res.error || "Failed to save progress");
    }
    setIsCompleting(false);
  };

  if (isLoading || !sign) return <LoadingScreen />;

  return (
    <div className="min-h-screen bg-[#F4EDE4] text-[#2C3E33]">
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b-4 border-border/50 px-4 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <GameButton
              variant="back"
              onClick={() => router.push(`/lessons?category=${sign.category}`)}
            >
              <ArrowLeft size={24} strokeWidth={3} />
            </GameButton>
            <div>
              <h1 className="font-display text-2xl font-black text-primary uppercase tracking-tighter leading-none">
                {sign.title}
              </h1>
              <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest mt-1">
                Category: {sign.category}
              </p>
            </div>
          </div>
          <CoinDisplay amount={dashboard?.coins || 0} />
        </div>
      </nav>

      <main className="pt-24 pb-12 px-6 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
          <div className="lg:col-span-7 h-[500px] lg:h-[700px]">
            <Sign3DViewer
              modelUrl={sign.model_url}
              animationName={sign.animation_name}
            />
          </div>

          <div className="lg:col-span-5 space-y-4">
            <div className="bg-white rounded-[2.5rem] p-8 border-b-8 border-slate-200 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <div className="bg-primary/10 px-4 py-1 rounded-full border-2 border-primary/20">
                  <span className="text-primary font-black text-xs uppercase tracking-widest">
                    {sign.difficulty}
                  </span>
                </div>
                <div className="flex gap-1">
                  {[1, 2, 3].map((s) => (
                    <Star
                      key={s}
                      size={18}
                      className={cn(
                        isMastered
                          ? "fill-yellow-400 text-yellow-500"
                          : "text-slate-200",
                      )}
                    />
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-6">
                <div className="w-48 h-24 bg-slate-50 rounded-3xl border-4 border-slate-100 flex items-center justify-center shadow-inner">
                  <span className="text-6xl font-black text-primary">
                    {sign.nepali_char}
                  </span>
                </div>
                <h2 className="text-5xl font-black uppercase tracking-tighter">
                  {sign.title}
                </h2>
                <p className="text-muted-foreground font-medium leading-relaxed italic">
                  Study the instructor's hand movements. When you're ready,
                  enter the Practice Room.
                </p>
              </div>
              <div className="pt-4 border-t border-slate-100 space-y-4">
                <p className="text-muted-foreground font-medium leading-relaxed italic">
                  {typeof sign.description === "object" &&
                  sign.description !== null
                    ? sign.description.text
                    : sign.description ||
                      `Analyze the instructor's hand movements for "${sign.nepali_char}". Rotate the view to see the sign from different angles.`}
                </p>

                {sign.description?.image_url && (
                  <div className="relative group/img">
                    <div className="absolute inset-0 bg-primary/10 rounded-2xl blur-xl opacity-0 group-hover/img:opacity-100 transition-opacity" />
                    <div className="relative bg-slate-50 rounded-2xl border-2 border-slate-100 p-2 overflow-hidden shadow-sm">
                      <p className="text-[8px] font-black uppercase text-muted-foreground mb-1 ml-1 tracking-widest">
                        Reference Image
                      </p>
                      <img
                        src={`http://localhost:8000${sign.description.image_url}`}
                        alt="Sign Reference"
                        className="w-full h-auto rounded-xl object-cover"
                        onError={(e) =>
                          (e.currentTarget.style.display = "none")
                        }
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white rounded-[2rem] p-6 border-b-8 border-slate-200 shadow-xl">
              <h4 className="font-black uppercase text-[10px] tracking-widest text-muted-foreground text-center mb-6">
                Completion Rewards
              </h4>
              <div className="flex justify-around items-center">
                <RewardIcon
                  icon={<Zap />}
                  label="+50 XP"
                  color="bg-yellow-400"
                />
                <RewardIcon
                  icon={<ShieldCheck />}
                  label="MASTERED"
                  color="bg-primary"
                />
                <RewardIcon icon={<Trophy />} label="BADGE" color="bg-accent" />
              </div>
            </div>

            <div className="space-y-4">
              {isMastered && (
                <div className="flex item-center animate-in zoom-in fade-in duration-500">
                  <GameButton
                    variant="duolingo"
                    size="lg"
                    className="w-full py-2 text-2xl flex items-center gap-3 whitespace-nowrap"
                    onClick={() => router.push(`/practice/${id}`)}
                  >
                    <Video size={28} className="animate-pulse shrink-0" />
                    <span>READY TO PRACTICE</span>
                  </GameButton>
                </div>
              )}

              {/* 2. COMPLETE BUTTON - Changes to a 'Success' label once done */}
              <GameButton
                variant={isMastered ? "retro" : "retro"}
                size="lg"
                className={cn(
                  "w-full py-2 text-2xl shadow-xl transition-all",
                  isMastered && "opacity-60 saturate-50 cursor-default",
                )}
                onClick={handleComplete}
                isLoading={isCompleting}
                disabled={isMastered}
              >
                <div className="flex items-center gap-4">
                  {isMastered ? (
                    <>
                      <CheckCircle2 size={32} className="text-white" />
                      LESSON COMPLETED
                    </>
                  ) : (
                    "COMPLETE LESSON !"
                  )}
                </div>
              </GameButton>

              {/* Small Hint for the user */}
              {!isMastered && (
                <p className="text-center text-[10px] font-black uppercase text-muted-foreground tracking-widest animate-pulse">
                  Complete the lesson to unlock practice mode
                </p>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// Fixed the RewardIcon helper to use standard cn
function RewardIcon({
  icon,
  label,
  color,
}: {
  icon: any;
  label: string;
  color: string;
}) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className={cn(
          "w-12 h-12 rounded-2xl flex items-center justify-center text-white border-b-4",
          color,
        )}
      >
        {icon}
      </div>
      <span className="font-black text-[9px] uppercase tracking-tighter">
        {label}
      </span>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background gap-4">
      <div className="w-16 h-16 border-8 border-primary/20 border-t-primary rounded-full animate-spin" />
      <p className="font-black uppercase tracking-widest text-primary animate-pulse">
        Entering Dojo...
      </p>
    </div>
  );
}
