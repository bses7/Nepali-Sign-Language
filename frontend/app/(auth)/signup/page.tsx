"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { GameButton } from "@/components/game-button";
import { GameHeader } from "@/components/game-header";
import { useAuthStore } from "@/lib/store/auth";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { FcGoogle } from "react-icons/fc";
import { FaGithub } from "react-icons/fa";
import {
  Mail,
  Lock,
  Sparkles,
  Fingerprint,
  GraduationCap,
  Presentation,
  Phone,
  User,
} from "lucide-react";

export default function SignupPage() {
  const router = useRouter();
  const { signup, isLoading, error, isAuthenticated, clearError } =
    useAuthStore();

  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    phoneNumber: "",
    role: "student",
  });
  const [localError, setLocalError] = useState("");

  useEffect(() => {
    if (isAuthenticated) router.push("/dashboard");
  }, [isAuthenticated, router]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError("");
    clearError();

    if (formData.password !== formData.confirmPassword) {
      setLocalError("Passwords do not match");
      return;
    }

    const success = await signup(
      formData.email,
      formData.password,
      formData.firstName,
      formData.lastName,
      formData.phoneNumber,
      formData.role,
    );

    if (success) {
      toast.success("Account created! Please sign in.");
      router.push("/login");
    } else {
      setLocalError(error || "Signup failed");
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background px-4 py-20 relative overflow-hidden">
      {/* 1. FLOATING DECORATIVE ELEMENTS */}
      <div className="absolute inset-0 pointer-events-none">
        <Sparkles
          className="absolute top-[10%] left-[15%] text-primary/20 animate-float"
          size={40}
        />
        <GraduationCap
          className="absolute bottom-[15%] left-[10%] text-secondary/20 animate-wiggle"
          size={60}
        />
        <Fingerprint
          className="absolute top-[20%] right-[10%] text-accent/20 animate-float"
          size={50}
        />
        <Presentation
          className="absolute bottom-[10%] right-[15%] text-primary/15 animate-wiggle"
          size={45}
        />

        {/* Glows */}
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-secondary/5 rounded-full blur-[120px]" />
      </div>

      <div className="relative w-full max-w-lg">
        <GameHeader
          title="SignLearn"
          subtitle="Choose your path!"
          variant="duolingo"
        />

        <div className="glass rounded-[3rem] p-8 md:p-10 border-b-[12px] border-border/40 shadow-2xl mt-6 space-y-8">
          <div className="space-y-2 text-center">
            <h2 className="font-display text-3xl font-black uppercase tracking-tighter text-foreground">
              Create Profile
            </h2>
            <p className="text-muted-foreground text-[10px] font-black uppercase tracking-[0.2em]">
              Step 1: Select your role
            </p>
          </div>

          {(localError || error) && (
            <div className="bg-destructive/10 border-b-4 border-destructive text-destructive px-6 py-3 rounded-2xl text-sm font-bold animate-wiggle text-center">
              {localError || error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* ROLE SELECTION */}
            <div className="grid grid-cols-2 gap-4">
              <button
                type="button"
                onClick={() => setFormData({ ...formData, role: "student" })}
                className={cn(
                  "flex flex-col items-center gap-2 p-5 rounded-3xl border-2 transition-all duration-200",
                  formData.role === "student"
                    ? "bg-primary text-white border-primary shadow-[0_8px_0_0_#4a5f4b] -translate-y-1"
                    : "bg-input border-border border-b-8 hover:border-primary/50 text-muted-foreground",
                )}
              >
                <GraduationCap
                  size={32}
                  className={
                    formData.role === "student" ? "text-white" : "text-primary"
                  }
                />
                <span className="font-black uppercase text-xs">Student</span>
              </button>

              <button
                type="button"
                onClick={() => setFormData({ ...formData, role: "teacher" })}
                className={cn(
                  "flex flex-col items-center gap-2 p-5 rounded-3xl border-2 transition-all duration-200",
                  formData.role === "teacher"
                    ? "bg-secondary text-foreground border-secondary shadow-[0_8px_0_0_#829480] -translate-y-1"
                    : "bg-input border-border border-b-8 hover:border-secondary/50 text-muted-foreground",
                )}
              >
                <Presentation
                  size={32}
                  className={
                    formData.role === "teacher"
                      ? "text-foreground"
                      : "text-secondary"
                  }
                />
                <span className="font-black uppercase text-xs">Teacher</span>
              </button>
            </div>

            {/* INPUTS SECTION */}
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="relative">
                  <User
                    className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/50"
                    size={18}
                  />
                  <input
                    name="firstName"
                    placeholder="First Name"
                    value={formData.firstName}
                    onChange={handleChange}
                    className="game-input pl-12"
                    required
                  />
                </div>
                <div className="relative">
                  <User
                    className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/50"
                    size={18}
                  />
                  <input
                    name="lastName"
                    placeholder="Last Name"
                    value={formData.lastName}
                    onChange={handleChange}
                    className="game-input pl-12"
                    required
                  />
                </div>
              </div>

              <div className="relative">
                <Mail
                  className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/50"
                  size={18}
                />
                <input
                  name="email"
                  type="email"
                  placeholder="Email Address"
                  value={formData.email}
                  onChange={handleChange}
                  className="game-input pl-12"
                  required
                />
              </div>

              <div className="relative">
                <Phone
                  className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/50"
                  size={18}
                />
                <input
                  name="phoneNumber"
                  type="tel"
                  placeholder="Phone Number"
                  value={formData.phoneNumber}
                  onChange={handleChange}
                  className="game-input pl-12"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="relative">
                  <Lock
                    className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/50"
                    size={18}
                  />
                  <input
                    name="password"
                    type="password"
                    placeholder="Password"
                    value={formData.password}
                    onChange={handleChange}
                    className="game-input pl-12"
                    required
                  />
                </div>
                <div className="relative">
                  <Lock
                    className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/50"
                    size={18}
                  />
                  <input
                    name="confirmPassword"
                    type="password"
                    placeholder="Confirm"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    className="game-input pl-12"
                    required
                  />
                </div>
              </div>
            </div>

            <GameButton
              type="submit"
              variant="retro"
              isLoading={isLoading}
              className="w-full mt-2"
            >
              Sign Up
            </GameButton>
          </form>

          {/* SOCIALS */}
          <div className="flex items-center gap-4">
            <div className="flex-1 h-1 bg-border/30 rounded-full" />
            <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">
              Or
            </span>
            <div className="flex-1 h-1 bg-border/30 rounded-full" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <button
              type="button"
              onClick={() =>
                (window.location.href =
                  "http://localhost:8000/api/v1/auth/login/google")
              }
              className="flex items-center justify-center gap-2 bg-white border-b-4 border-gray-200 active:border-b-0 active:translate-y-1 py-3 rounded-2xl font-bold transition-all hover:bg-gray-50 text-sm"
            >
              <FcGoogle className="text-xl" />
              Google
            </button>

            <button
              type="button"
              onClick={() =>
                (window.location.href =
                  "http://localhost:8000/api/v1/auth/login/github")
              }
              className="flex items-center justify-center gap-2 bg-slate-900 text-white border-b-4 border-slate-700 active:border-b-0 active:translate-y-1 py-3 rounded-2xl font-bold transition-all hover:bg-slate-800 text-sm"
            >
              <FaGithub className="text-xl" />
              GitHub
            </button>
          </div>

          <p className="text-center text-sm font-bold text-muted-foreground">
            Already a learner?{" "}
            <Link
              href="/login"
              className="text-primary hover:underline underline-offset-4"
            >
              Sign In
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
