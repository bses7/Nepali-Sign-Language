"use client";

import {
  useEffect,
  useState,
  Suspense,
  useRef,
  memo,
  useCallback,
} from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { generationService } from "@/lib/api/generation";
import { avatarService } from "@/lib/api/avatar";
import { GameButton } from "@/components/game-button";
import { ParkEnv } from "@/components/park-env";
import {
  Sparkles,
  Languages,
  Cpu,
  ArrowLeft,
  Loader2,
  HelpCircle,
  X,
  BookOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { Canvas } from "@react-three/fiber";
import {
  PerspectiveCamera,
  Environment,
  ContactShadows,
  useGLTF,
  useAnimations,
  Float,
  PresentationControls,
} from "@react-three/drei";
import * as THREE from "three";

// Fixed Scene3D type definition to include 'mode'
const Scene3D = memo(
  ({
    avatarFolder,
    resultUrl,
    mode, // Added back
    onAnimationFinished,
  }: {
    avatarFolder: string;
    resultUrl: string | null;
    mode: string; // Added back
    onAnimationFinished: () => void;
  }) => {
    return (
      <Canvas shadows dpr={[1, 2]}>
        <PerspectiveCamera makeDefault position={[0, 3, 14]} fov={45} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[10, 20, 10]} intensity={2} castShadow />

        <Suspense fallback={null}>
          <ParkEnv />
          <Float speed={1.5} rotationIntensity={0.1} floatIntensity={0.2}>
            <TrainingInstructor
              folder={avatarFolder || "avatar"}
              activeModelUrl={resultUrl}
              defaultAnimation="Idle"
              onAnimationFinished={onAnimationFinished}
            />
          </Float>
          <Environment preset="park" />
          <ContactShadows
            position={[0, -1.99, 0]}
            opacity={0.5}
            scale={15}
            blur={2.5}
            far={10}
          />
        </Suspense>
      </Canvas>
    );
  },
);

Scene3D.displayName = "Scene3D";

function TrainingInstructor({
  folder,
  defaultAnimation,
  activeModelUrl,
  onAnimationFinished,
}: {
  folder: string;
  defaultAnimation: string;
  activeModelUrl: string | null;
  onAnimationFinished: () => void;
}) {
  const group = useRef<THREE.Group>(null);
  const modelUrl =
    activeModelUrl || avatarService.getAnimationUrl(folder, defaultAnimation);
  const { scene, animations } = useGLTF(modelUrl) as any;
  const { actions, mixer } = useAnimations(animations, group);
  const [scale, setScale] = useState(5);

  const handleWheel = (e: any) => {
    e.stopPropagation();
    const delta = e.deltaY * -0.001;
    setScale((prev) => Math.min(Math.max(prev + delta, 1.5), 8));
  };

  useEffect(() => {
    if (scene) {
      scene.position.set(0, 0, 0);
      scene.rotation.set(0, 0, 0);
      scene.scale.set(1, 1, 1);
      scene.traverse((obj: any) => {
        if (obj.isMesh) {
          obj.castShadow = true;
          obj.receiveShadow = true;
        }
      });
    }
  }, [scene]);

  useEffect(() => {
    if (actions) {
      Object.values(actions).forEach((action) => action?.stop());
      const animationKeys = Object.keys(actions);
      if (animationKeys.length > 0) {
        const action = actions[animationKeys[0]]!;
        if (activeModelUrl) {
          action.setLoop(THREE.LoopOnce, 1);
          action.clampWhenFinished = true;
          const onFinish = () => {
            onAnimationFinished();
            mixer.removeEventListener("finished", onFinish);
          };
          mixer.addEventListener("finished", onFinish);
        } else {
          action.setLoop(THREE.LoopRepeat, Infinity);
        }
        action.reset().fadeIn(0.3).play();
      }
    }
    return () => {
      mixer.removeEventListener("finished", onAnimationFinished);
    };
  }, [actions, activeModelUrl, mixer, onAnimationFinished]);

  return (
    <PresentationControls
      global={false}
      cursor={true}
      snap={true}
      speed={2}
      zoom={1}
      polar={[-Math.PI / 10, Math.PI / 10]}
      azimuth={[-Infinity, Infinity]}
    >
      <group
        ref={group}
        onWheel={handleWheel}
        scale={[scale, scale, scale]}
        position={[0, -2.5, 0]}
      >
        <primitive object={scene} />
      </group>
    </PresentationControls>
  );
}

export default function GenerationPage() {
  const router = useRouter();
  const { isAuthenticated, dashboard, fetchDashboard } = useAuthStore();
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [inputText, setInputText] = useState("");
  const [isSkeletonEnabled, setIsSkeletonEnabled] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [skeletonVideoUrl, setSkeletonVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!isAuthenticated) router.push("/login");
    fetchDashboard();
  }, [isAuthenticated, router, fetchDashboard]);

  const handleAnimationFinished = useCallback(() => {
    setResultUrl(null);
    setSkeletonVideoUrl(null);
  }, []);

  const handleGenerate = async () => {
    if (!inputText) return toast.error("Enter text to translate!");

    setIsGenerating(true);
    setResultUrl(null);
    setSkeletonVideoUrl(null);

    try {
      const avatarPromise = generationService.generateSign(inputText);
      const skeletonPromise = isSkeletonEnabled
        ? generationService.generateSkeleton(inputText)
        : Promise.resolve(null);

      const [avatarRes, skeletonRes] = await Promise.all([
        avatarPromise,
        skeletonPromise,
      ]);

      const baseUrl =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      let nextAvatarUrl = null;
      let nextSkeletonUrl = null;

      if (avatarRes?.success && avatarRes.data) {
        nextAvatarUrl = avatarRes.data.model_url;
        if (nextAvatarUrl.startsWith("/"))
          nextAvatarUrl = `${baseUrl}${nextAvatarUrl}`;
      }

      if (skeletonRes?.success && skeletonRes.data) {
        nextSkeletonUrl = skeletonRes.data.video_url;
        if (nextSkeletonUrl.startsWith("/"))
          nextSkeletonUrl = `${baseUrl}${nextSkeletonUrl}`;
      }

      if (nextAvatarUrl) {
        setResultUrl(nextAvatarUrl);
        if (isSkeletonEnabled && nextSkeletonUrl) {
          setSkeletonVideoUrl(nextSkeletonUrl);
        }
        toast.success("Ready!");
      } else {
        toast.error("Avatar node failed.");
      }
    } catch (error) {
      console.error("Generation Error:", error);
      toast.error("AI node timeout.");
    } finally {
      setIsGenerating(false);
    }
  };

  if (!dashboard) return null;

  return (
    <div className="h-screen w-full bg-slate-950 overflow-hidden relative font-sans">
      <div className="absolute inset-0 z-0">
        <Scene3D
          avatarFolder={dashboard.equipped_avatar_folder || "avatar"}
          resultUrl={resultUrl}
          mode="sign"
          onAnimationFinished={handleAnimationFinished}
        />
      </div>

      {skeletonVideoUrl && resultUrl && (
        <div className="hidden md:block absolute top-32 right-10 z-[60] w-72 aspect-video bg-black rounded-3xl border-4 border-blue-500 shadow-2xl overflow-hidden animate-pop-spin">
          {/* ... rest of the video code ... */}
          <video
            src={skeletonVideoUrl}
            autoPlay
            loop
            muted
            playsInline
            className="w-full h-full object-contain"
          />
        </div>
      )}

      <div className="absolute top-10 left-20 z-50 hidden md:block">
        <GameButton
          variant="back"
          onClick={() => router.push("/dashboard")}
          className="shadow-2xl"
        >
          <ArrowLeft size={24} strokeWidth={3} />
        </GameButton>
      </div>

      <div className="absolute top-10 left-1/2 -translate-x-1/2 z-50 flex p-1 bg-black/40 backdrop-blur-xl rounded-2xl border border-white/10 shadow-2xl">
        <button className="flex items-center gap-2 px-6 py-2 rounded-xl font-black text-[10px] uppercase tracking-widest transition-all bg-primary text-white shadow-lg cursor-default">
          <Sparkles size={14} /> 3D Avatar
        </button>

        {/* Add hidden md:flex to this button */}
        <button
          onClick={() => {
            setIsSkeletonEnabled(!isSkeletonEnabled);
            setSkeletonVideoUrl(null);
          }}
          className={cn(
            "hidden md:flex items-center gap-2 px-6 py-2 rounded-xl font-black text-[10px] uppercase tracking-widest transition-all",
            isSkeletonEnabled
              ? "bg-blue-600 text-white shadow-lg"
              : "text-white/40 hover:text-white",
          )}
        >
          <Cpu size={14} /> Skeleton View {isSkeletonEnabled ? "ON" : "OFF"}
        </button>
      </div>

      <div className="absolute bottom-0 left-0 right-0 z-50 p-6 md:p-12 flex flex-col items-center">
        <div className="w-full max-w-4xl bg-white rounded-[2.5rem] md:rounded-[3rem] p-4 md:p-6 border-b-[12px] border-slate-200 shadow-2xl flex flex-col md:flex-row gap-4 md:gap-6 items-center">
          <div className="relative flex-1 w-full">
            <BookOpen
              className="absolute left-4 top-1/2 -translate-y-1/2 text-primary"
              size={20}
            />
            <input
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Type in Nepali or English..."
              className="game-input pl-12 pr-4 h-14 md:h-16"
            />
          </div>
          <GameButton
            variant="duolingo"
            className="w-full md:w-[180px] py-3 text-lg uppercase tracking-tight"
            onClick={handleGenerate}
            isLoading={isGenerating}
          >
            Translate
          </GameButton>
        </div>
      </div>

      {/* --- HELP TRIGGER BUTTON --- */}
      <div className="absolute bottom-10 right-10 z-50">
        <button
          onClick={() => setIsHelpOpen(true)}
          className="w-16 h-16 bg-yellow-400 border-b-8 border-yellow-600 rounded-full flex items-center justify-center text-white shadow-2xl hover:scale-110 active:translate-y-2 active:border-b-0 transition-all group"
        >
          <HelpCircle
            size={32}
            className="group-hover:rotate-12 transition-transform"
          />
        </button>
      </div>

      {/* --- NEURAL MANUAL MODAL --- */}
      {isHelpOpen && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center p-6 bg-black/60 backdrop-blur-md">
          <div className="bg-white rounded-[3rem] w-full max-w-lg p-10 border-b-[12px] border-slate-200 shadow-2xl relative animate-pop-spin overflow-y-auto max-h-[90vh] custom-scrollbar">
            {/* Close Button */}
            <button
              onClick={() => setIsHelpOpen(false)}
              className="absolute top-4 right-4 bg-red-500 text-white p-2 rounded-xl border-b-4 border-red-700 active:translate-y-1 active:border-b-0 transition-all z-[10000]"
            >
              <X size={24} />
            </button>

            <div className="text-center space-y-8">
              <div className="inline-block bg-blue-100 p-5 rounded-[2rem] border-b-4 border-blue-200 text-blue-600">
                <Cpu size={48} className="animate-pulse" />
              </div>

              <h3 className="font-display text-4xl font-black uppercase tracking-tighter text-foreground leading-none">
                Generation Guide
              </h3>

              <div className="space-y-6 text-left">
                <div className="space-y-3">
                  <p className="text-[10px] font-black uppercase tracking-[0.2em] text-primary ml-2">
                    Translation Protocol
                  </p>

                  <div className="flex gap-4 p-4 bg-slate-50 rounded-2xl border-b-4 border-slate-100">
                    <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-black shadow-sm">
                      1
                    </div>
                    <p className="text-sm font-bold text-muted-foreground leading-snug">
                      Enter any{" "}
                      <span className="text-primary">Nepali or English</span>{" "}
                      word into the neural console below.
                    </p>
                  </div>

                  <div className="flex gap-4 p-4 bg-slate-50 rounded-2xl border-b-4 border-slate-100">
                    <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-black shadow-sm">
                      2
                    </div>
                    <div className="flex flex-col gap-1">
                      <p className="text-sm font-bold text-muted-foreground leading-snug">
                        Choose your output mode:
                      </p>
                      <ul className="text-[10px] font-bold text-slate-500 list-disc ml-4 space-y-1">
                        <li>
                          <span className="text-primary uppercase">
                            3D Avatar:
                          </span>{" "}
                          Full visual demonstration.
                        </li>
                        <li>
                          <span className="text-blue-500 uppercase">
                            Skeleton:
                          </span>{" "}
                          Technical bone-structure video.
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <p className="text-[10px] font-black uppercase tracking-[0.2em] text-orange-500 ml-2">
                    Inspection
                  </p>
                  <div className="grid grid-cols-1 gap-3">
                    <div className="flex items-center gap-4 p-3 bg-orange-50 rounded-2xl border-b-4 border-orange-100">
                      <div className="p-2 bg-white rounded-xl shadow-sm">
                        <Languages size={16} className="text-orange-600" />
                      </div>
                      <p className="text-xs font-bold text-orange-900 leading-tight">
                        The AI nodes will process your request and materialize
                        the sign in real-time.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <GameButton
                variant="duolingo"
                className="w-full py-6 text-2xl shadow-xl"
                onClick={() => setIsHelpOpen(false)}
              >
                Start Generating
              </GameButton>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

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
