"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { GameButton } from "@/components/game-button";
import { GameHeader } from "@/components/game-header";
import { apiClient } from "@/lib/api/client";
import {
  Video,
  Type,
  FileText,
  Upload,
  Zap,
  Coins,
  CheckCircle2,
  X,
  Fingerprint,
  Sparkles,
  GraduationCap,
  Lock,
  HelpCircle,
  Lightbulb,
  MousePointer2,
  ShieldCheck,
  Mail,
  ChevronLeft,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { teachersService } from "@/lib/api/teacher";

export default function TeacherContributePage() {
  const router = useRouter();
  const { dashboard, fetchDashboard, isAuthenticated } = useAuthStore();
  const [isUploading, setIsUploading] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);

  const [isHelpOpen, setIsHelpOpen] = useState(false);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }

    if (dashboard) {
      const isAuthorized =
        dashboard.role === "teacher" || dashboard.role === "admin";

      if (!isAuthorized) {
        router.push("/dashboard");
        return;
      }
    }

    fetchDashboard();
  }, [isAuthenticated, dashboard?.role, router]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("video/")) {
        return toast.error("Invalid file type. Please upload a video.");
      }
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
    }
  };

  const removeVideo = () => {
    setVideoFile(null);
    setVideoPreview(null);
  };

  const [showVerifiedError, setShowVerifiedError] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const canUpload =
      dashboard?.is_verified_teacher || dashboard?.role === "admin";

    const titleValue = (
      e.currentTarget.elements.namedItem("title") as HTMLInputElement
    ).value;
    const descValue = (
      e.currentTarget.elements.namedItem("description") as HTMLTextAreaElement
    ).value;

    if (!canUpload) {
      toast.error("Restricted Access", {
        description: "Only verified teachers can upload new signs.",
        icon: <Lock size={20} className="text-destructive" />,
      });

      setShowVerifiedError(true);
      setTimeout(() => setShowVerifiedError(false), 4000);
      return;
    }

    if (!videoFile) return toast.error("Please select a video file!");

    const formData = new FormData();
    formData.append("title", titleValue);
    formData.append("description", descValue);
    formData.append("video", videoFile);
    setIsUploading(true);

    try {
      const response = await teachersService.uploadSign(formData);

      if (response.success) {
        toast.success("Contribution Received!", {
          description: "Your video has been sent for review. +200 XP rewarded!",
        });
        await fetchDashboard();
        router.push("/dashboard");
      } else {
        toast.error(response.error || "Upload failed");
      }
    } catch (err) {
      toast.error("An error occurred during upload.");
    } finally {
      setIsUploading(false);
    }
  };

  if (
    !dashboard ||
    (dashboard.role !== "teacher" && dashboard.role !== "admin")
  )
    return null;

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background px-4 py-20 relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <Sparkles
          className="absolute top-[15%] left-[10%] text-primary/20 animate-float"
          size={40}
        />
        <Fingerprint
          className="absolute bottom-[20%] left-[15%] text-secondary/20 animate-wiggle"
          size={50}
        />
        <GraduationCap
          className="absolute top-[25%] right-[12%] text-accent/20 animate-float"
          size={45}
        />
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-secondary/5 rounded-full blur-[120px]" />
      </div>

      <main className="max-w-6xl mx-auto space-y-8">
        <GameHeader
          title="SignLearn"
          subtitle="Content Creator Studio"
          variant="duolingo"
        />

        <div className="bg-white rounded-[3.5rem] p-8 md:p-12 border-b-[16px] border-slate-200 shadow-2xl relative overflow-hidden">
          <div className="mb-10 space-y-1">
            <h2 className="text-4xl font-black uppercase tracking-tighter">
              Submit New Sign
            </h2>
            <p className="text-muted-foreground font-bold">
              Help us grow the library by recording high-quality sign
              demonstrations.
            </p>
          </div>

          <form
            onSubmit={handleSubmit}
            className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start"
          >
            <div className="space-y-6">
              <div className="space-y-2">
                <label className="text-[10px] font-black uppercase tracking-[0.3em] text-primary ml-2">
                  Sign Name
                </label>
                <div className="relative">
                  <Type
                    className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/40"
                    size={20}
                  />
                  <input
                    name="title"
                    placeholder="e.g. NAMASTE"
                    className="game-input pl-12"
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-[10px] font-black uppercase tracking-[0.3em] text-primary ml-2">
                  Instructions
                </label>
                <div className="relative">
                  <FileText
                    className="absolute left-4 top-5 text-muted-foreground/40"
                    size={20}
                  />
                  <textarea
                    name="description"
                    placeholder="Describe how to perform this sign step-by-step..."
                    className="game-input pl-12 min-h-[280px] resize-none leading-relaxed"
                    required
                  />
                </div>
              </div>
            </div>

            {/* --- RIGHT SIDE: VIDEO CAPTURE --- */}
            <div className="flex flex-col h-full">
              <label className="text-[10px] font-black uppercase tracking-[0.3em] text-primary ml-2 mb-2">
                Video Demonstration
              </label>

              <div
                className={cn(
                  "relative flex-1 min-h-[300px] lg:min-h-full rounded-[2.5rem] border-4 border-dashed transition-all overflow-hidden flex flex-col items-center justify-center",
                  videoPreview
                    ? "border-primary bg-black"
                    : "border-slate-200 bg-slate-50 hover:border-primary/50 hover:bg-primary/5",
                )}
              >
                {videoPreview ? (
                  <>
                    <video
                      src={videoPreview}
                      className="w-full h-full object-cover"
                      controls
                    />
                    <button
                      type="button"
                      onClick={removeVideo}
                      className="absolute top-4 right-4 bg-red-500 text-white p-2 rounded-xl border-b-4 border-red-700 active:translate-y-1 active:border-b-0"
                    >
                      <X size={20} />
                    </button>
                  </>
                ) : (
                  <div className="text-center p-10">
                    <div className="w-20 h-20 bg-white rounded-3xl border-b-4 border-slate-200 flex items-center justify-center mx-auto mb-6">
                      <Video className="text-primary" size={32} />
                    </div>
                    <p className="font-black uppercase text-sm tracking-tight text-slate-600">
                      Video Demonstration
                    </p>
                    <p className="text-xs text-muted-foreground mt-2 max-w-[200px] mx-auto">
                      Upload the video of you performing the sign clearly.
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      MP4, MOV or AVI accepted
                    </p>
                    <input
                      type="file"
                      name="video"
                      accept="video/*"
                      onChange={handleFileChange}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      required
                    />
                  </div>
                )}
              </div>
            </div>

            <div className="lg:col-span-2 pt-6 border-t-2 border-slate-50 flex flex-col md:flex-row items-center justify-between gap-8">
              <div className="flex items-center gap-6 bg-primary/5 px-8 py-4 rounded-[2rem] border-2 border-primary/10">
                <div className="flex items-center gap-2">
                  <Zap className="text-yellow-500 fill-yellow-500" size={20} />
                  <span className="font-black text-sm uppercase">+200 XP</span>
                </div>
                <div className="w-px h-6 bg-primary/20" />
                <div className="flex items-center gap-2">
                  <Coins
                    className="text-yellow-500 fill-yellow-500"
                    size={20}
                  />
                  <span className="font-black text-sm uppercase">
                    +100 Coins
                  </span>
                </div>
                <div className="w-px h-6 bg-primary/20" />
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="text-primary" size={20} />
                  <span className="font-black text-[10px] uppercase text-muted-foreground">
                    Certified Contribution
                  </span>
                </div>
              </div>

              <GameButton
                variant="retro"
                size="md"
                type="submit"
                className="w-full md:w-auto p-2 text-3xl"
                isLoading={isUploading}
              >
                UPLOAD TO ACADEMY
              </GameButton>
            </div>
          </form>
        </div>
      </main>

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

      {isHelpOpen && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-md">
          <div className="bg-white rounded-[3rem] w-full max-w-lg p-7 border-b-[12px] border-slate-200 shadow-2xl relative animate-pop-spin overflow-y-auto max-h-[90vh] custom-scrollbar">
            <button
              onClick={() => setIsHelpOpen(false)}
              className="absolute top-5 right-5 bg-red-500 text-white p-2 rounded-xl border-b-4 border-red-700 active:translate-y-1 active:border-b-0 transition-all z-[10000]"
            >
              <X size={24} />
            </button>

            <div className="text-center space-y-6">
              <div className="inline-block bg-yellow-100 p-5 rounded-[2rem] border-b-4 border-yellow-200">
                <GraduationCap size={48} className="text-yellow-600" />
              </div>

              <h3 className="font-display text-4xl font-black uppercase tracking-tighter text-foreground leading-none">
                Academy Guide
              </h3>

              <div className="space-y-6 text-left">
                <div className="space-y-3">
                  <p className="text-[10px] font-black uppercase tracking-[0.2em] text-primary ml-2">
                    Submitting Signs
                  </p>
                  <div className="flex gap-4 p-4 bg-slate-50 rounded-2xl border-b-4 border-slate-100">
                    <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-black">
                      1
                    </div>
                    <p className="text-sm font-bold text-muted-foreground leading-tight">
                      Enter the <span className="text-primary">Sign Name</span>{" "}
                      and a clear description of the movements.
                    </p>
                  </div>
                  <div className="flex gap-4 p-4 bg-slate-50 rounded-2xl border-b-4 border-slate-100">
                    <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-black">
                      2
                    </div>
                    <p className="text-sm font-bold text-muted-foreground leading-tight">
                      Upload a high-quality{" "}
                      <span className="text-blue-500">Video Demo</span>. Ensure
                      your hands are clearly visible.
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-[10px] font-black uppercase tracking-[0.2em] text-blue-600 ml-2">
                    Teacher Verification
                  </p>
                  <div className="bg-blue-50 p-6 rounded-[2rem] border-b-8 border-blue-100 space-y-4">
                    <div className="flex items-center gap-3">
                      <ShieldCheck className="text-blue-600" size={24} />
                      <h4 className="font-black uppercase text-sm text-blue-900">
                        Unlock Full Access
                      </h4>
                    </div>
                    <p className="text-xs font-bold text-blue-800 leading-relaxed">
                      To become a verified instructor and earn{" "}
                      <span className="text-yellow-600">XP rewards</span>,
                      please send your <span className="underline">CV</span> and
                      a verification request to:
                    </p>
                    <div className="bg-white p-3 rounded-xl border-2 border-blue-200 flex items-center justify-center gap-2 group cursor-pointer hover:border-blue-400 transition-all">
                      <Mail size={16} className="text-blue-600" />
                      <span className="font-black text-blue-600 text-sm select-all">
                        signlearn@gmail.com
                      </span>
                    </div>
                    <p className="text-[9px] font-black uppercase text-blue-400 text-center tracking-widest">
                      Approval takes 24-48 hours
                    </p>
                  </div>
                </div>
              </div>

              <GameButton
                variant="retro"
                className="w-full py-2 text-2xl shadow-xl"
                onClick={() => setIsHelpOpen(false)}
              >
                READY TO Contribute!
              </GameButton>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
