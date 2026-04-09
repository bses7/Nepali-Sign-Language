"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { adminService } from "@/lib/api/admin";
import { GameButton } from "@/components/game-button";
import { CoinDisplay } from "@/components/game-stats";
import { ProfileDropdown } from "@/components/profile-dropdown";
import {
  ShieldCheck,
  UserCheck,
  ShieldAlert,
  Mail,
  ChevronLeft,
  Search,
  Users,
  BadgeCheck,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

export default function AdminTeachersPage() {
  const router = useRouter();
  const { isAuthenticated, dashboard, fetchDashboard } = useAuthStore();
  const [teachers, setTeachers] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isActionLoading, setIsActionLoading] = useState<number | null>(null);

  useEffect(() => {
    if (!isAuthenticated) return router.push("/login");
    if (dashboard && dashboard.role !== "admin")
      return router.push("/dashboard");

    loadTeachers();
  }, [isAuthenticated, dashboard]);

  const loadTeachers = async () => {
    setIsLoading(true);
    const res = await adminService.getTeachersList();
    if (res.success) setTeachers(res.data || []);
    setIsLoading(false);
  };

  const handleVerify = async (userId: number) => {
    setIsActionLoading(userId);
    const res = await adminService.verifyTeacher(userId);
    if (res.success) {
      toast.success("Instructor Verified", {
        description: "Email notification transmitted.",
      });
      // Update local state instead of full re-fetch
      setTeachers((prev) =>
        prev.map((t) =>
          t.id === userId ? { ...t, is_verified_teacher: true } : t,
        ),
      );
    } else {
      toast.error(res.error || "Verification failed");
    }
    setIsActionLoading(null);
  };

  if (isLoading) return <LoadingScreen />;

  return (
    <div className="min-h-screen w-full bg-[#F4EDE4] text-[#2C3E33]">
      {/* HUD NAV */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b-4 border-border/50 px-4 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <GameButton
              variant="back"
              onClick={() => router.push("/dashboard")}
            >
              <ChevronLeft size={24} strokeWidth={3} />
            </GameButton>
            <h1 className="font-display text-2xl font-black text-primary tracking-tighter uppercase">
              Admin Console
            </h1>
          </div>
          <ProfileDropdown userName={dashboard?.first_name || "Admin"} />
        </div>
      </nav>

      <main className="pt-28 pb-12 px-4 max-w-5xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="bg-white rounded-[3rem] p-8 border-b-[12px] border-slate-200 shadow-xl flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-6">
            <div className="w-20 h-20 bg-blue-500 rounded-3xl border-b-8 border-blue-700 flex items-center justify-center text-white shadow-lg">
              <Users size={40} />
            </div>
            <div>
              <h2 className="text-4xl font-black uppercase tracking-tighter leading-none">
                Instructor Registry
              </h2>
              <p className="text-muted-foreground font-bold uppercase text-[10px] tracking-widest mt-2">
                Authorization Management
              </p>
            </div>
          </div>
          <div className="bg-slate-50 px-6 py-3 rounded-2xl border-2 border-slate-100 flex items-center gap-3">
            <span className="text-3xl font-black text-primary">
              {teachers.length}
            </span>
            <span className="text-[10px] font-black uppercase text-muted-foreground leading-tight">
              Total
              <br />
              Teachers
            </span>
          </div>
        </div>

        {/* TEACHERS LIST */}
        <div className="bg-white rounded-[3rem] p-6 md:p-10 border-b-[16px] border-slate-200 shadow-2xl space-y-4">
          {teachers.length > 0 ? (
            teachers.map((teacher) => (
              <div
                key={teacher.id}
                className={cn(
                  "flex flex-col md:flex-row items-center justify-between p-6 rounded-[2.5rem] border-2 transition-all gap-6",
                  teacher.is_verified_teacher
                    ? "bg-primary/5 border-primary/20"
                    : "bg-slate-50 border-slate-100 border-dashed",
                )}
              >
                {/* Teacher Info */}
                <div className="flex items-center gap-5 flex-1">
                  <div className="w-14 h-14 rounded-2xl bg-white border-b-4 border-slate-200 flex items-center justify-center font-black text-xl text-primary shadow-sm">
                    {teacher.first_name.charAt(0)}
                  </div>
                  <div>
                    <h3 className="font-black text-xl uppercase tracking-tight flex items-center gap-2">
                      {teacher.first_name} {teacher.last_name}
                      {teacher.is_verified_teacher}
                    </h3>
                    <div className="flex items-center gap-3 text-muted-foreground">
                      <span className="flex items-center gap-1 text-[10px] font-bold">
                        <Mail size={12} /> {teacher.email}
                      </span>
                      <div className="w-1 h-1 bg-slate-300 rounded-full" />
                      <span className="text-[10px] font-bold uppercase">
                        Lvl {teacher.level || 1}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-6">
                  <div className="text-right hidden sm:block">
                    <p className="text-[9px] font-black uppercase text-muted-foreground tracking-widest">
                      Status
                    </p>
                    <p
                      className={cn(
                        "font-black text-xs uppercase",
                        teacher.is_verified_teacher
                          ? "text-primary"
                          : "text-yellow-600",
                      )}
                    >
                      {teacher.is_verified_teacher
                        ? "Verified"
                        : "Pending Review"}
                    </p>
                  </div>

                  {teacher.is_verified_teacher ? (
                    <div className="bg-primary/20 p-3 rounded-2xl text-primary border-b-4 border-primary/30">
                      <ShieldCheck size={28} />
                    </div>
                  ) : (
                    <GameButton
                      variant="retro"
                      size="md"
                      className="p-2"
                      onClick={() => handleVerify(teacher.id)}
                      isLoading={isActionLoading === teacher.id}
                    >
                      VERIFY Teacher
                    </GameButton>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="py-20 text-center space-y-4">
              <ShieldAlert size={64} className="mx-auto text-slate-200" />
              <p className="font-bold text-muted-foreground">
                No instructors found in the registry.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-[#F4EDE4] gap-4">
      <div className="w-16 h-16 border-8 border-primary/20 border-t-primary rounded-full animate-spin" />
      <p className="font-black uppercase tracking-widest text-primary animate-pulse">
        Accessing Admin Crypt...
      </p>
    </div>
  );
}
