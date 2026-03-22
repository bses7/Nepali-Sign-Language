"use client";

import { useEffect, useRef, useState, Suspense } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { lessonsService } from "@/lib/api/lessons";
import { GameButton } from "@/components/game-button";
import { ClassroomEnv } from "@/components/classroom-env";
import {
  CheckCircle2,
  ArrowLeft,
  Play,
  Pause,
  Lightbulb,
  HelpCircle,
  RefreshCcw,
  Target,
  Trophy,
  Timer,
  AlertTriangle,
  Info,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  Float,
  ContactShadows,
  Environment,
  useGLTF,
  useAnimations,
  PerspectiveCamera,
  PresentationControls,
} from "@react-three/drei";
import * as THREE from "three";
import { practiceService } from "@/lib/api/practice";

function Teacher({ url, animationName, speed, isPaused, scale = 4 }: any) {
  const group = useRef<THREE.Group>(null);
  const fullUrl = url.startsWith("http") ? url : `http://localhost:8000${url}`;
  const { scene, animations } = useGLTF(fullUrl) as any;
  const { actions } = useAnimations(animations, group);

  useEffect(() => {
    const action = actions[animationName] || Object.values(actions)[0];
    if (action) {
      action.reset().fadeIn(0.5).play();
    }
    return () => {
      action?.fadeOut(0.5);
    };
  }, [actions, animationName]);

  useFrame(() => {
    const action = actions[animationName] || Object.values(actions)[0];
    if (action) {
      action.paused = isPaused;
      action.setEffectiveTimeScale(speed);
    }
  });

  return (
    <primitive
      ref={group}
      object={scene}
      scale={scale}
      // CHANGE: Set Z to 0. Only keep Y as -2 to put feet on the floor.
      position={[0, -2, 0]}
      castShadow
    />
  );
}

export default function PracticePage() {
  const { id } = useParams();
  const router = useRouter();
  const { isAuthenticated, fetchDashboard } = useAuthStore();

  const [sign, setSign] = useState<any>(null);
  const [status, setStatus] = useState<
    "ready" | "practicing" | "completed" | "failed"
  >("ready");
  const INITIAL_FEEDBACK = {
    progress: 0,
    confidence: 0,
    prediction: "...",
  };

  const [feedback, setFeedback] = useState({
    progress: 0,
    confidence: 0,
    prediction: "...",
  });
  const [report, setReport] = useState<any>(null);

  const [timeLeft, setTimeLeft] = useState(40);
  const [isPaused, setIsPaused] = useState(false);
  const [speed, setSpeed] = useState(0.5);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const frameIntervalRef = useRef<any>(null);
  const timerIntervalRef = useRef<any>(null);

  useEffect(() => {
    if (!isAuthenticated) return router.push("/login");
    lessonsService.getSignById(id as string).then((res) => {
      if (res.success) setSign(res.data);
    });
  }, [id, isAuthenticated, router]);

  useEffect(() => {
    if (status === "practicing" && timeLeft > 0) {
      timerIntervalRef.current = setInterval(() => {
        setTimeLeft((prev) => prev - 1);
      }, 1000);
    } else if (timeLeft === 0 && status === "practicing") {
      stopPractice("failed");
      toast.error("Time's up! Try again.");
    }

    return () => clearInterval(timerIntervalRef.current);
  }, [status, timeLeft]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault();
        setIsPaused((prev) => !prev);
      }

      if (e.code === "Enter") {
        if (status === "ready" || status === "failed") {
          startPractice();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [status, sign]);

  const startPractice = async () => {
    setFeedback(INITIAL_FEEDBACK);

    if (!sign) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      if (videoRef.current) videoRef.current.srcObject = stream;

      const wsUrl = practiceService.getWebSocketUrl(sign.nepali_char);
      socketRef.current = new WebSocket(wsUrl);

      socketRef.current.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === "final_report") {
          setReport(data.report);
          stopPractice("completed");
        } else {
          setFeedback({
            progress: data.progress,
            confidence: data.confidence,
            prediction: data.prediction,
          });
        }
      };

      setTimeLeft(40);
      setStatus("practicing");
      frameIntervalRef.current = setInterval(sendFrame, 120);
    } catch (err) {
      toast.error("Camera required for Dojo entry!");
    }
  };

  const sendFrame = () => {
    if (
      socketRef.current?.readyState === WebSocket.OPEN &&
      videoRef.current &&
      canvasRef.current
    ) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, 320, 240);
        socketRef.current.send(canvasRef.current.toDataURL("image/jpeg", 0.4));
      }
    }
  };

  const stopPractice = (nextStatus: any) => {
    clearInterval(frameIntervalRef.current);
    clearInterval(timerIntervalRef.current);
    socketRef.current?.close();
    const stream = videoRef.current?.srcObject as MediaStream;
    stream?.getTracks().forEach((track) => track.stop());

    setStatus(nextStatus);

    if (nextStatus === "failed" || nextStatus === "ready") {
      setFeedback(INITIAL_FEEDBACK);
    }
  };

  const claimRewards = async () => {
    setIsSaving(true);
    try {
      const res = await practiceService.saveResults(100);
      console.log("Reward Response:", res);

      if (res.success) {
        toast.success("Practice Rewards Claimed!");
        await fetchDashboard();
        router.push(`/lessons?category=${sign.category}`);
      } else {
        toast.error(res.error || "Failed to claim rewards");
      }
    } catch (err) {
      toast.error("An error occurred while claiming rewards");
    } finally {
      setIsSaving(false);
    }
  };

  const [windowWidth, setWindowWidth] = useState(
    typeof window !== "undefined" ? window.innerWidth : 1200,
  );
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const isMobile = windowWidth < 768;
  const [localZoom, setLocalZoom] = useState(1);
  const handleWheel = (e: React.WheelEvent) => {
    setLocalZoom((prev) =>
      Math.min(Math.max(prev - e.deltaY * 0.001, 0.6), 1.8),
    );
  };

  if (!sign) return null;

  return (
    <div
      onWheel={handleWheel}
      className="h-screen w-full bg-slate-950 overflow-hidden relative touch-none select-none font-sans"
    >
      <div className="absolute inset-0 z-0">
        <Canvas shadows>
          <PerspectiveCamera
            makeDefault
            position={[-0.2, 2, isMobile ? 18 : 12]}
            fov={45}
          />
          <ambientLight intensity={0.6} />
          <spotLight
            position={[20, 30, 10]}
            angle={0.2}
            penumbra={1}
            intensity={2.5}
            castShadow
          />
          <Suspense fallback={null}>
            <ClassroomEnv targetChar={sign.nepali_char} />
            <group position={[0, 0, -3]} scale={localZoom}>
              <PresentationControls
                global={false}
                cursor={true}
                snap={false}
                speed={3}
                zoom={1}
                rotation={[0, 0, 0]}
                polar={[-0.1, 0.1]}
                azimuth={[-Math.PI / 2, Math.PI / 2]}
              >
                <Float speed={1.5} rotationIntensity={0.1} floatIntensity={0.1}>
                  <Teacher
                    url={sign.model_url}
                    animationName={sign.animation_name}
                    speed={speed}
                    isPaused={isPaused}
                    scale={isMobile ? 4.5 : 4}
                  />
                </Float>
              </PresentationControls>
            </group>

            <Environment preset="apartment" />
          </Suspense>
        </Canvas>
      </div>

      {/* 2. HUD NAV & TIMER */}
      <nav className="absolute top-0 left-8 right-0 z-50 flex justify-between items-start p-6 md:p-10 pointer-events-none">
        <div className="pointer-events-auto">
          <GameButton
            variant="back"
            onClick={() => router.push(`/lessons?category=${sign.category}`)}
          >
            <ArrowLeft size={24} strokeWidth={3} />
          </GameButton>
        </div>

        {status === "practicing" && (
          <div className="pointer-events-auto bg-white/10 backdrop-blur-md border-2 border-white/20 px-6 py-3 rounded-3xl flex items-center gap-4 animate-bounce-slow">
            <Timer
              className={cn(
                "transition-colors",
                timeLeft < 10 ? "text-red-500 animate-pulse" : "text-white",
              )}
            />
            <span
              className={cn(
                "text-3xl font-black tabular-nums",
                timeLeft < 10 ? "text-red-500" : "text-white",
              )}
            >
              {timeLeft}s
            </span>
          </div>
        )}
      </nav>

      {/* 3. BOTTOM UI CONSOLE */}
      <div className="absolute bottom-0 left-0 right-0 z-50 p-4 md:p-10 flex flex-col items-center">
        {status === "ready" ||
        status === "practicing" ||
        status === "failed" ? (
          <div
            className={cn(
              "w-full max-w-6xl flex items-center md:items-end gap-4 md:gap-10 transition-all duration-700",
              isMobile ? "flex-col-reverse" : "flex-row justify-between",
            )}
          >
            {/* PLAYBACK CONTROLS */}
            <div className="bg-white rounded-[2rem] md:rounded-[3rem] p-4 md:p-6 border-b-[8px] border-slate-200 flex items-center gap-4 shadow-2xl">
              <button
                onClick={() => setIsPaused(!isPaused)}
                className="w-12 h-12 md:w-16 md:h-16 bg-primary rounded-full flex items-center justify-center text-white active:scale-90 border-b-4 border-green-800 shadow-lg"
              >
                {isPaused ? (
                  <Play fill="currentColor" />
                ) : (
                  <Pause fill="currentColor" />
                )}
              </button>
              <div className="flex gap-2">
                {[0.5, 1].map((s) => (
                  <button
                    key={s}
                    onClick={() => setSpeed(s)}
                    className={cn(
                      "px-4 py-2 rounded-xl font-black text-xs border-b-4 transition-all",
                      speed === s
                        ? "bg-primary border-green-800 text-white"
                        : "bg-slate-100 border-slate-200 text-slate-400",
                    )}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>

            {/* WEBCAM MONITOR - Dynamic Height */}
            <div className="relative group">
              <div
                className={cn(
                  "aspect-video rounded-[2.5rem] border-4 md:border-8 overflow-hidden shadow-2xl transition-all duration-500 relative",
                  status === "practicing"
                    ? isMobile
                      ? "w-[300px] h-[220px]"
                      : "w-[500px] h-[350px]"
                    : isMobile
                      ? "w-[240px] h-[150px]"
                      : "w-[450px] h-[225px]",
                  status === "practicing"
                    ? "border-primary bg-black"
                    : "border-white/10 bg-slate-900",
                )}
              >
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover scale-x-[-1]"
                />
                <canvas
                  ref={canvasRef}
                  width="320"
                  height="240"
                  className="hidden"
                />

                {(status === "ready" || status === "failed") && (
                  <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex flex-col items-center justify-center gap-4">
                    {status === "failed" && (
                      <div className="bg-red-500/20 text-red-400 p-3 rounded-2xl flex items-center gap-2 mb-2">
                        <AlertTriangle size={20} />
                        <span className="font-black uppercase text-xs">
                          Session Expired
                        </span>
                      </div>
                    )}
                    <GameButton
                      variant="jelly"
                      size="md"
                      onClick={startPractice}
                    >
                      {status === "failed" ? "Retry Trial" : "Start Camera"}
                    </GameButton>
                  </div>
                )}

                {/* HOLD INDICATOR OVERLAY */}
                {feedback.progress > 0 && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none bg-primary/10">
                    <div className="bg-white px-6 py-3 rounded-full shadow-2xl border-b-4 border-slate-200 animate-pulse flex items-center gap-3">
                      <div className="w-6 h-6 rounded-full border-4 border-primary border-t-transparent animate-spin" />
                      <span className="font-black text-primary uppercase tracking-widest text-sm">
                        Hold Steady!
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* ACCURACY HUD */}
            <div
              className={cn(
                "bg-white rounded-[2rem] md:rounded-[3.5rem] p-6 md:p-8 border-b-[8px] md:border-b-[12px] border-slate-200 shadow-2xl text-center min-w-[200px]",
                isMobile ? "flex items-center gap-6 py-4" : "space-y-2",
              )}
            >
              <div>
                <p className="flex justify-center text-[10px] font-black uppercase text-muted-foreground tracking-widest leading-none mb-1">
                  <Target size={16} className="text-primary mr-1" />
                  Confidence
                </p>
                <div className="text-3xl md:text-5xl font-black text-primary tabular-nums tracking-tighter">
                  {(feedback.confidence * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="w-full max-w-lg bg-white rounded-[3.5rem] p-8 md:p-10 border-b-[16px] border-slate-200 shadow-2xl space-y-6 animate-in zoom-in-95 duration-500 text-slate-900 mb-10 overflow-y-auto max-h-[80vh]">
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center border-b-4 border-green-200">
                <CheckCircle2 size={40} className="text-green-600" />
              </div>
              <h2 className="text-4xl font-black uppercase tracking-tighter">
                Excellent!
              </h2>
              <div className="flex gap-2">
                <span className="px-4 py-1 bg-primary/10 text-primary rounded-full text-xs font-black uppercase">
                  Score: {report?.accuracy_percentage}%
                </span>
                <span className="px-4 py-1 bg-yellow-100 text-yellow-700 rounded-full text-xs font-black uppercase">
                  {report?.status}
                </span>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-slate-50 p-4 rounded-2xl border-l-8 border-primary shadow-sm">
                <div className="flex items-center gap-2 mb-3 text-primary">
                  <Info size={18} />
                  <span className="font-black uppercase text-xs">Feedback</span>
                </div>
                <ul className="space-y-2">
                  {report?.feedback?.map((tip: string, idx: number) => (
                    <li
                      key={idx}
                      className="flex gap-3 text-sm font-bold text-slate-600 leading-tight"
                    >
                      <span className="text-primary">•</span>
                      {tip}
                    </li>
                  )) || (
                    <li className="text-sm font-bold text-slate-600 italic">
                      "Perfect form! No adjustments needed."
                    </li>
                  )}
                </ul>
              </div>

              <div className="bg-slate-50 p-5 rounded-2xl border-b-4 border-slate-100 flex justify-between items-center">
                <div className="flex items-center gap-3 font-black uppercase text-xs">
                  <Trophy className="text-yellow-500" /> Practice Reward
                </div>
                <span className="font-black text-2xl text-green-600">
                  +100 XP
                </span>
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <GameButton
                variant="duolingo"
                className="w-full py-2 text-2xl shadow-[0_12px_0_0_#d97900]"
                onClick={claimRewards}
                isLoading={isSaving}
              >
                CLAIM REWARDS 💰
              </GameButton>

              <button
                onClick={() => {
                  setReport(null);
                  setStatus("ready");
                  setTimeLeft(40);
                  setFeedback(INITIAL_FEEDBACK);
                }}
                className="w-full py-4 font-black uppercase text-xs text-slate-400 hover:text-primary transition-colors flex items-center justify-center gap-2"
              >
                <RefreshCcw size={16} /> Try for Higher Score
              </button>
            </div>
          </div>
        )}
      </div>

      {/* HELP BUTTON */}
      <div className="absolute bottom-10 right-10 z-50">
        <button
          onClick={() => setIsHelpOpen(true)}
          className="w-16 h-16 bg-primary border-b-8 border-green-900 rounded-full flex items-center justify-center text-white shadow-2xl hover:scale-110 active:translate-y-2 transition-all"
        >
          <HelpCircle size={32} />
        </button>
      </div>

      <div className="absolute bottom-10 right-10 z-50">
        <button
          onClick={() => setIsHelpOpen(true)}
          className="w-16 h-16 bg-primary border-b-8 border-green-900 rounded-full flex items-center justify-center text-white shadow-2xl hover:scale-110 active:translate-y-2 active:border-b-0 transition-all group"
        >
          <HelpCircle
            size={32}
            className="group-hover:rotate-12 transition-transform"
          />
        </button>
      </div>

      {isHelpOpen && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center p-6 bg-black/70 backdrop-blur-md animate-in fade-in duration-300">
          <div className="bg-white rounded-[3rem] w-full max-w-lg p-10 border-b-[12px] border-slate-200 shadow-2xl relative animate-pop-spin max-h-[90vh] overflow-y-auto custom-scrollbar">
            <div className="text-center space-y-8">
              <div className="inline-block bg-yellow-100 p-5 rounded-[2rem] border-b-4 border-yellow-200">
                <Lightbulb size={48} className="text-yellow-500" />
              </div>

              <h3 className="font-display text-4xl font-black uppercase tracking-tighter text-foreground leading-none">
                Dojo Protocol
              </h3>

              <div className="space-y-6 text-left">
                <div className="space-y-3">
                  <p className="text-[10px] font-black uppercase tracking-[0.2em] text-primary ml-2">
                    Mission
                  </p>
                  <div className="flex gap-4 p-4 bg-slate-50 rounded-2xl border-b-4 border-slate-100">
                    <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-black">
                      1
                    </div>
                    <p className="text-sm font-bold text-muted-foreground leading-snug">
                      Watch the{" "}
                      <span className="text-primary font-black">
                        Instructor
                      </span>{" "}
                      carefully. Use the speed buttons to see complex finger
                      movements.
                    </p>
                  </div>
                  <div className="flex gap-4 p-4 bg-slate-50 rounded-2xl border-b-4 border-slate-100">
                    <div className="flex-shrink-0 w-8 h-8 bg-accent text-white rounded-full flex items-center justify-center font-black">
                      2
                    </div>
                    <p className="text-sm font-bold text-muted-foreground leading-snug">
                      Activate your camera and match the pose. Hold the sign for{" "}
                      <span className="text-accent font-black">3 seconds</span>{" "}
                      until the system authenticates.
                    </p>
                  </div>
                </div>
              </div>

              <GameButton
                variant="duolingo"
                className="w-full py-2 text-2xl"
                onClick={() => setIsHelpOpen(false)}
              >
                BEGIN TRIAL
              </GameButton>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
