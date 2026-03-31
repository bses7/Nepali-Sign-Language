"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";
import { User, LogOut, Bell, ChevronLeft } from "lucide-react";
import { usersService } from "@/lib/api/users";
import { cn } from "@/lib/utils";

interface ProfileDropdownProps {
  userName: string;
}

export const ProfileDropdown = ({ userName }: ProfileDropdownProps) => {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [notifications, setNotifications] = useState<any[]>([]);

  const dropdownRef = useRef<HTMLDivElement>(null);
  const { logout } = useAuthStore();

  // 1. FETCH NOTIFICATIONS ON MOUNT (To show the dot immediately)
  useEffect(() => {
    const fetchNotifs = () => {
      usersService.getNotifications().then((res) => {
        if (res.success) setNotifications(res.data || []);
      });
    };
    fetchNotifs();
  }, []);

  // Calculate unread count
  const unreadCount = notifications.filter((n) => !n.is_read).length;

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
        setShowNotifications(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleRead = async (id: number) => {
    const res = await usersService.markNotificationRead(id);
    if (res.success) {
      setNotifications((prev) =>
        prev.map((n) => (n.id === id ? { ...n, is_read: true } : n)),
      );
    }
  };

  const handleLogout = () => {
    logout();
    setIsOpen(false);
    router.push("/login");
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "relative h-12 w-12 flex items-center justify-center transition-all duration-200 cursor-pointer focus:outline-none",
          "bg-white rounded-2xl  hover:bg-slate-50 active:translate-y-1 active:border-b-0",
          isOpen && "translate-y-1 border-b-0",
        )}
      >
        <div className="h-9 w-9 rounded-3xl bg-linear-to-br from-primary to-secondary flex items-center justify-center text-white font-black text-lg shadow-inner ring-2 ring-white/50">
          {userName.charAt(0).toUpperCase()}
        </div>

        {unreadCount > 0 && (
          <div className="absolute -top-1.5 -right-1.5 flex h-5 w-5 z-10">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-orange-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-5 w-5 bg-orange-500 border-2 border-white items-center justify-center">
              <span className="text-[10px] text-white font-black leading-none">
                {unreadCount}
              </span>
            </span>
          </div>
        )}
      </button>

      {/* --- DROPDOWN (Style preserved exactly as requested) --- */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 glass rounded-2xl border-2 border-primary/20 shadow-xl overflow-hidden z-50 animate-in fade-in slide-in-from-top-2">
          <div className="px-4 py-4 border-b border-border/50 bg-muted/30 flex items-center justify-between">
            {!showNotifications ? (
              <>
                <div className="truncate pr-2">
                  <p className="font-semibold text-foreground truncate text-sm">
                    {userName}
                  </p>
                  <p className="text-[10px] text-muted-foreground uppercase font-bold tracking-widest">
                    Learner
                  </p>
                </div>
                <button
                  onClick={() => setShowNotifications(true)}
                  className="p-1.5 hover:bg-primary/10 rounded-lg text-muted-foreground hover:text-primary transition-colors relative"
                >
                  <Bell size={18} />
                  {unreadCount > 0 && (
                    <span className="absolute top-1 right-1 w-2 h-2 bg-orange-500 rounded-full border border-white" />
                  )}
                </button>
              </>
            ) : (
              <button
                onClick={() => setShowNotifications(false)}
                className="flex items-center gap-2 text-xs font-bold text-primary hover:opacity-80 transition-opacity"
              >
                <ChevronLeft size={16} />
                Back to Menu
              </button>
            )}
          </div>

          <div className="py-2">
            {!showNotifications ? (
              <>
                <button
                  onClick={() => {
                    router.push("/profile");
                    setIsOpen(false);
                  }}
                  className="w-full px-4 py-3 text-left text-foreground hover:bg-primary/10 transition-colors flex items-center gap-3 group"
                >
                  <User className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                  <span className="font-medium text-sm">My Profile</span>
                </button>

                <div className="h-px bg-border/50 my-1" />

                <button
                  onClick={handleLogout}
                  className="w-full px-4 py-3 text-left text-destructive hover:bg-destructive/10 transition-colors flex items-center gap-3 group"
                >
                  <LogOut className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                  <span className="font-medium text-sm">Sign Out</span>
                </button>
              </>
            ) : (
              <div className="max-h-[160px] overflow-y-auto custom-scrollbar px-2 space-y-1">
                {notifications.length > 0 ? (
                  notifications.map((n) => (
                    <div
                      key={n.id}
                      onClick={() => !n.is_read && handleRead(n.id)}
                      className={cn(
                        "p-3 rounded-xl border transition-all cursor-pointer flex gap-3",
                        n.is_read
                          ? "bg-slate-50/50 border-transparent opacity-60"
                          : "bg-primary/5 border-primary/20 shadow-sm",
                      )}
                    >
                      <div className="flex-1">
                        <p
                          className={cn(
                            "text-[11px] leading-tight",
                            n.is_read
                              ? "text-muted-foreground"
                              : "text-foreground font-bold",
                          )}
                        >
                          {n.message}
                        </p>
                        <p className="text-[9px] mt-1 text-muted-foreground opacity-50 font-medium">
                          Just now
                        </p>
                      </div>
                      {!n.is_read && (
                        <div className="w-1.5 h-1.5 bg-primary rounded-full mt-1.5 shrink-0" />
                      )}
                    </div>
                  ))
                ) : (
                  <div className="py-10 text-center opacity-30 flex flex-col items-center">
                    <Bell size={24} />
                    <p className="text-[10px] font-bold uppercase mt-2">
                      All caught up
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProfileDropdown;
