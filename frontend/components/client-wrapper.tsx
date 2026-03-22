"use client";

import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { Gamepad2 } from "lucide-react";
import { Toaster } from "sonner";
import { useAuthStore } from "@/lib/store/auth";
import { MaintenanceScreen } from "@/components/maintenance-screen";

export function ClientWrapper({ children }: { children: React.ReactNode }) {
  const isOffline = useAuthStore((state) => state.isOffline);

  if (isOffline) {
    return <MaintenanceScreen />;
  }

  return (
    <SidebarProvider defaultOpen={false}>
      <AppSidebar />
      <main className="relative flex-1">
        {/* Global Hamburger Button */}
        <div className="fixed top-4 left-4 z-60">
          <SidebarTrigger className="group h-12 w-12 bg-white border-slate-200 transition-all hover:bg-slate-50 active:translate-y-1 active:border-b-0">
            <Gamepad2
              size={24}
              className="text-primary group-hover:rotate-12 group-hover:scale-150 transition-transform duration-200"
            />
          </SidebarTrigger>
        </div>

        {children}

        <Toaster
          position="top-center"
          toastOptions={{
            unstyled: true,
            classNames: {
              toast:
                "w-full flex items-center gap-3 p-4 rounded-2xl border-b-4 shadow-2xl transition-all duration-300",
              success: "bg-[#F4EDE4] border-primary text-primary",
              error: "bg-[#FDF2F2] border-destructive text-destructive",
              info: "bg-white border-blue-500 text-blue-600",
              title: "font-black uppercase tracking-tight text-sm",
              description: "font-bold text-xs opacity-80",
            },
          }}
        />
      </main>
    </SidebarProvider>
  );
}
