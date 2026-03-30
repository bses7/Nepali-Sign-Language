"use client";

import { useEffect, useState, Suspense, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { quizService } from "@/lib/api/quiz";
import { GameButton } from "@/components/game-button";
import { Sign3DViewer } from "@/components/sign-3d-viewer";
import {
  Trophy,
  BrainCircuit,
  Clock,
  CheckCircle2,
  Zap,
  Loader2,
  Flame,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

function QuizContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const category = searchParams.get("category") || "vowel";
  const difficulty = searchParams.get("difficulty") || "easy";

  const { fetchDashboard } = useAuthStore();

  // Quiz State
  const [quizData, setQuizData] = useState<any>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [isAnswered, setIsAnswered] = useState(false);
  const [status, setStatus] = useState<"loading" | "playing" | "finished">(
    "loading",
  );
  const [isSubmitting, setIsSubmitting] = useState(false); // New state for backend call

  // Timer State
  const [timeLeft, setTimeLeft] = useState(30);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    quizService.generateQuiz(category, difficulty).then((res) => {
      if (res.success) {
        setQuizData(res.data);
        setStatus("playing");
      } else {
        toast.error("Failed to generate quiz");
        router.push("/lessons");
      }
    });
  }, [category, difficulty, router]);

  // TIMER LOGIC
  useEffect(() => {
    if (status === "playing" && timeLeft > 0) {
      timerRef.current = setInterval(() => {
        setTimeLeft((prev) => prev - 1);
      }, 1000);
    } else if (timeLeft === 0 && status === "playing") {
      // Auto-finish if time runs out
      if (timerRef.current) clearInterval(timerRef.current);
      setStatus("finished");
      toast.error("TIME EXPIRED!");
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [status, timeLeft]);

  const handleAnswer = (optionId: number) => {
    if (isAnswered) return;

    setSelectedId(optionId);
    setIsAnswered(true);

    const isCorrect =
      optionId === quizData.questions[currentIndex].correct_sign_id;
    if (isCorrect) {
      setScore((prev) => prev + 1);
      toast.success("Point Captured!", { icon: "🎯" });
    } else {
      toast.error("Incorrect!");
    }

    setTimeout(() => {
      if (currentIndex < quizData.questions.length - 1) {
        setCurrentIndex((prev) => prev + 1);
        setSelectedId(null);
        setIsAnswered(false);
      } else {
        if (timerRef.current) clearInterval(timerRef.current);
        setStatus("finished");
      }
    }, 1500);
  };

  // --- NEW: LOGIC TO SUBMIT AND REDIRECT ---
  const handleClaimReward = async () => {
    setIsSubmitting(true);
    try {
      const res = await quizService.submitQuiz({
        score: score,
        category,
        difficulty,
      });

      if (res.success) {
        toast.success("Claimed Quiz Rewards!");
        await fetchDashboard();
        router.push(`/lessons?category=${category}`);
      } else {
        toast.error("Failed to claim rewards.");
      }
    } catch (error) {
      toast.error("Server connection error.");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (status === "loading")
    return <LoadingScreen message="Calibrating Arena..." />;

  const currentQuestion = quizData.questions[currentIndex];

  return (
    <div className="min-h-screen bg-slate-950 text-white overflow-hidden relative font-sans">
      {/* 1. DANGER OVERLAY */}
      {timeLeft <= 10 && status === "playing" && (
        <div className="absolute inset-0 border-[10px] border-red-500/20 pointer-events-none z-[60] animate-pulse" />
      )}

      {/* 2. HUD TOP NAVIGATION */}
      <nav className="absolute top-0 left-0 right-0 z-50 p-8 flex justify-between items-center pointer-events-none">
        <div className="flex flex-col gap-4 pointer-events-auto">
          <div className="bg-black/60 backdrop-blur-xl px-6 py-2 rounded-2xl border border-white/10 shadow-2xl flex items-center gap-4">
            <div className="flex flex-col">
              <span className="text-[8px] font-black uppercase text-primary tracking-[0.3em]">
                Quiz
              </span>
              <span className="text-white font-black text-xs uppercase">
                {difficulty} {category}
              </span>
            </div>
            <div className="w-px h-6 bg-white/10" />
            <div className="flex flex-col">
              <span className="text-[8px] font-black uppercase text-primary tracking-[0.3em]">
                Progress
              </span>
              <span className="text-white font-black text-xs uppercase">
                {currentIndex + 1} / 3
              </span>
            </div>
          </div>
          <div className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className={cn(
                  "h-1.5 w-16 rounded-full transition-all duration-500",
                  i < currentIndex
                    ? "bg-primary"
                    : i === currentIndex
                      ? "bg-yellow-400 animate-pulse"
                      : "bg-white/10",
                )}
              />
            ))}
          </div>
        </div>

        <div
          className={cn(
            "pointer-events-auto flex flex-col items-center gap-1 transition-all duration-300",
            timeLeft <= 5 ? "scale-125" : "scale-100",
          )}
        >
          <div
            className={cn(
              "bg-black/60 backdrop-blur-xl px-8 py-3 rounded-full border-2 flex items-center gap-3 shadow-2xl",
              timeLeft <= 10
                ? "border-red-500 text-red-500"
                : "border-white/10 text-white",
            )}
          >
            <Clock size={20} className={cn(timeLeft <= 10 && "animate-spin")} />
            <span className="text-3xl font-black tabular-nums">
              {timeLeft}s
            </span>
          </div>
          <p
            className={cn(
              "text-[8px] font-black uppercase tracking-[0.5em]",
              timeLeft <= 10 ? "text-red-500" : "text-white/40",
            )}
          >
            Time Remaining
          </p>
        </div>

        <div className="bg-white p-4 rounded-[2.5rem] border-b-8 border-slate-200 shadow-2xl flex items-center gap-4 pointer-events-auto">
          <div className="bg-primary/10 p-2 rounded-xl">
            <BrainCircuit className="text-primary" />
          </div>
          <div className="text-right">
            <p className="text-[10px] font-black text-muted-foreground uppercase leading-none mb-1">
              Score
            </p>
            <p className="font-black text-2xl text-slate-800 leading-none tabular-nums">
              {score} / 3
            </p>
          </div>
        </div>
      </nav>

      {/* 3. ARENA CONTENT */}
      <main className="h-screen w-full flex flex-col items-center justify-center p-6 pt-20">
        {status === "playing" ? (
          <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            <div className="lg:col-span-7 h-[450px] md:h-[600px] relative group">
              <div className="absolute inset-0 bg-primary/5 rounded-[4rem] border-2 border-primary/20 blur-2xl animate-pulse" />
              <div className="relative h-full w-full rounded-[4rem] overflow-hidden border-4 border-white/10 bg-slate-900 shadow-inner">
                <Sign3DViewer
                  key={currentIndex}
                  modelUrl={currentQuestion.model_url}
                  animationName={currentQuestion.animation_name}
                />
                <div className="absolute inset-0 pointer-events-none opacity-10 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%),linear-gradient(90deg,rgba(255,255,255,0.05),rgba(255,255,255,0),rgba(255,255,255,0.05))] bg-[length:100%_4px,20px_100%]" />
              </div>
              <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 bg-primary px-8 py-2 rounded-2xl border-b-4 border-green-900 shadow-xl">
                <p className="text-[10px] font-black uppercase tracking-[0.4em] text-white">
                  Visual Feedback
                </p>
              </div>
            </div>

            <div className="lg:col-span-5 space-y-8">
              <div className="text-center lg:text-left space-y-2">
                <h3 className="text-4xl font-black uppercase tracking-tighter leading-none">
                  Identify Gesture
                </h3>
                <p className="text-white/40 font-bold text-sm">
                  Select the matching character from the data bank below.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-5">
                {currentQuestion.options.map((opt: any) => {
                  const isCorrect =
                    opt.sign_id === currentQuestion.correct_sign_id;
                  const isSelected = selectedId === opt.sign_id;
                  return (
                    <button
                      key={opt.sign_id}
                      onClick={() => handleAnswer(opt.sign_id)}
                      disabled={isAnswered}
                      className={cn(
                        "relative h-36 rounded-[2rem] flex items-center justify-center transition-all duration-150 border-b-[10px] active:translate-y-2 active:border-b-0",
                        !isAnswered &&
                          "bg-white border-slate-200 text-slate-800 hover:border-primary hover:-translate-y-1",
                        isAnswered &&
                          isCorrect &&
                          "bg-primary border-green-800 text-white scale-105 z-10",
                        isAnswered &&
                          isSelected &&
                          !isCorrect &&
                          "bg-red-500 border-red-800 text-white animate-wiggle",
                        isAnswered &&
                          !isSelected &&
                          "opacity-30 grayscale blur-[1px]",
                      )}
                    >
                      <span className="text-6xl font-black drop-shadow-md">
                        {opt.nepali_char}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        ) : (
          /* --- 4. FINAL RESULTS SCREEN --- */
          <div className="w-full max-w-md animate-in zoom-in-95 duration-500 relative">
            <div className="absolute -inset-4 bg-primary/20 blur-3xl rounded-full" />
            <div className="bg-white rounded-[4rem] p-10 border-b-[16px] border-slate-200 shadow-2xl text-slate-900 text-center space-y-8 relative">
              <div className="relative">
                <div className="w-24 h-24 bg-yellow-400 rounded-[2rem] flex items-center justify-center border-b-8 border-yellow-600 mx-auto rotate-3 animate-float">
                  <Trophy size={48} className="text-white fill-white" />
                </div>
              </div>

              <div className="space-y-1">
                <h2 className="text-4xl font-black uppercase tracking-tighter leading-none">
                  Evaluation Over
                </h2>
                <p className="text-muted-foreground font-black uppercase text-[10px] tracking-widest text-primary">
                  Unit Synchronized
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-50 p-5 rounded-3xl border-b-4 border-slate-100 flex flex-col items-center">
                  <span className="text-[8px] font-black uppercase text-slate-400 mb-1">
                    Final Score
                  </span>
                  <span className="text-4xl font-black text-foreground">
                    {score}/3
                  </span>
                </div>
                <div className="bg-slate-50 p-5 rounded-3xl border-b-4 border-slate-100 flex flex-col items-center">
                  <span className="text-[8px] font-black uppercase text-slate-400 mb-1">
                    Time Left
                  </span>
                  <span className="text-4xl font-black text-primary">
                    {timeLeft}s
                  </span>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-3 bg-primary/5 p-4 rounded-2xl border-2 border-primary/10">
                  <Flame className="text-orange-500" />
                  <div className="text-left leading-tight">
                    <p className="text-xs font-black uppercase">Rewards</p>
                    <p className="text-[10px] font-bold text-muted-foreground">
                      +{score * 50} XP • +{score * 20} Coins Earned
                    </p>
                  </div>
                </div>
              </div>

              <GameButton
                variant="retro" // Bright green for success
                className="w-full py-2 text-2xl shadow-xl"
                onClick={handleClaimReward}
                isLoading={isSubmitting}
              >
                CLAIM REWARD 💰
              </GameButton>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// ... Loading and Page wrapper logic remains the same ...

function LoadingScreen({ message }: { message: string }) {
  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-slate-950 gap-4">
      <Loader2 className="w-16 h-16 text-primary animate-spin" />
      <p className="font-black uppercase tracking-widest text-white animate-pulse">
        {message}
      </p>
    </div>
  );
}

export default function QuizPage() {
  return (
    <Suspense fallback={<LoadingScreen message="Entering Arena..." />}>
      <QuizContent />
    </Suspense>
  );
}
