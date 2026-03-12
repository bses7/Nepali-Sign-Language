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
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const items = [
  { title: "Dashboard", url: "/dashboard", icon: LayoutDashboard },
  { title: "Lessons", url: "/lessons", icon: BookOpen },
  { title: "Leaderboard", url: "/leaderboard", icon: Trophy },
  { title: "Shop", url: "/shop", icon: ShoppingBag },
];

export function AppSidebar() {
  const pathname = usePathname();

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
              {items.map((item) => {
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
        <button className="flex items-center gap-4 px-4 py-3 rounded-2xl font-black uppercase text-xs tracking-widest text-destructive hover:bg-destructive/10 transition-all">
          <LogOut size={20} />
          <span>Logout</span>
        </button>
      </SidebarFooter>
    </Sidebar>
  );
}
