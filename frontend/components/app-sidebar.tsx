"use client";

import * as React from "react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
} from "@/components/ui/sidebar";
import {
  LayoutDashboard,
  BookOpen,
  Trophy,
  ShoppingBag,
  LogOut,
  PlusCircle,
  ShieldCheck,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { cn } from "@/lib/utils";

export function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { logout, dashboard } = useAuthStore();

  const isTeacher = dashboard?.role === "teacher";
  const isAdmin = dashboard?.role === "admin";

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  const menuItems = [
    { title: "Dashboard", url: "/dashboard", icon: LayoutDashboard },
    { title: "Lessons", url: "/lessons", icon: BookOpen },
    { title: "Leaderboard", url: "/leaderboard", icon: Trophy },
    { title: "Shop", url: "/shop", icon: ShoppingBag },
  ];

  if (isTeacher || isAdmin) {
    menuItems.push({
      title: "Contribute",
      url: "/teacher/contribute",
      icon: PlusCircle,
    });
  }

  if (isAdmin) {
    menuItems.push({
      title: "Verify Teachers",
      url: "/admin/teachers",
      icon: ShieldCheck,
    });
  }

  return (
    <Sidebar className="border-r-8 border-border/50">
      <SidebarContent className="bg-white p-4">
        <SidebarGroup>
          <SidebarGroupLabel className="mt-24 mb-6 px-2">
            <span className="font-display text-3xl font-black text-primary tracking-tighter">
              SignLearn
            </span>
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="gap-2">
              {menuItems.map((item) => {
                const isActive = pathname === item.url;
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild size="lg">
                      <Link
                        href={item.url}
                        className={cn(
                          "flex items-center gap-4 px-4 py-6 rounded-2xl font-black uppercase text-xs tracking-widest transition-all border-b-4",
                          isActive
                            ? "bg-primary text-white border-primary-foreground/30 translate-y-[-2px] hover:bg-primary hover:text-white"
                            : "bg-white text-muted-foreground border-transparent hover:bg-slate-50 hover:border-slate-200",
                        )}
                      >
                        <item.icon size={24} />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="bg-white p-4 border-t border-border">
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-4 px-4 py-3 rounded-2xl font-black uppercase text-xs tracking-widest text-destructive hover:bg-destructive/10 transition-all cursor-pointer"
        >
          <LogOut size={20} />
          <span>Logout</span>
        </button>
      </SidebarFooter>
    </Sidebar>
  );
}
