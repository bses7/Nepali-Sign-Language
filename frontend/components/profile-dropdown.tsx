"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth";

interface ProfileDropdownProps {
  userName: string;
}

export const ProfileDropdown = ({ userName }: ProfileDropdownProps) => {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const { logout } = useAuthStore();

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLogout = () => {
    logout();
    setIsOpen(false);
    router.push("/login");
  };

  const handleProfile = () => {
    setIsOpen(false);
    router.push("/profile");
  };

  const handleSettings = () => {
    setIsOpen(false);
    // TODO: Navigate to settings page when created
    router.push("/settings");
  };

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Profile Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="h-10 w-10 rounded-full bg-linear-to-br from-primary to-secondary flex items-center justify-center text-white font-bold hover:shadow-lg hover:scale-110 transition-all duration-200 cursor-pointer"
        aria-label="User menu"
        aria-expanded={isOpen}
      >
        {userName.charAt(0).toUpperCase()}
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-56 glass rounded-2xl border-2 border-primary/20 shadow-xl overflow-hidden z-50 animate-in fade-in slide-in-from-top-2">
          {/* User Info */}
          <div className="px-4 py-4 border-b border-border/50">
            <p className="font-semibold text-foreground">{userName}</p>
            <p className="text-xs text-muted-foreground">Learner</p>
          </div>

          {/* Menu Items */}
          <div className="py-2">
            {/* Profile */}
            <button
              onClick={handleProfile}
              className="w-full px-4 py-3 text-left text-foreground hover:bg-primary/10 transition-colors flex items-center gap-3 group"
            >
              <span className="text-lg group-hover:scale-125 transition-transform">
                👤
              </span>
              <span className="font-medium">My Profile</span>
            </button>

            {/* Settings */}
            <button
              onClick={handleSettings}
              className="w-full px-4 py-3 text-left text-foreground hover:bg-primary/10 transition-colors flex items-center gap-3 group"
            >
              <span className="text-lg group-hover:scale-125 transition-transform">
                ⚙️
              </span>
              <span className="font-medium">Settings</span>
            </button>

            {/* Divider */}
            <div className="h-px bg-border/50 my-2" />

            {/* Logout */}
            <button
              onClick={handleLogout}
              className="w-full px-4 py-3 text-left text-destructive hover:bg-destructive/10 transition-colors flex items-center gap-3 group font-medium"
            >
              <span className="text-lg group-hover:scale-125 transition-transform">
                🚪
              </span>
              <span>Sign Out</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProfileDropdown;
