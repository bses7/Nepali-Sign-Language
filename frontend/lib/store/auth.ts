import { create } from "zustand";
import { persist } from "zustand/middleware";
import { authService } from "@/lib/api/auth";
import { usersService, DashboardData } from "@/lib/api/users";

// 1. Updated User interface to include new backend fields
export interface User {
  id?: string;
  email: string;
  first_name?: string;
  last_name?: string;
  phone_number?: string;
  role?: string;
  coins?: number;
}

export interface AuthStore {
  // Auth state
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Dashboard state
  dashboard: DashboardData | null;
  isDashboardLoading: boolean;

  // Auth actions
  login: (email: string, password: string) => Promise<boolean>;

  // 2. Updated signup signature to require all 6 fields
  signup: (
    email: string,
    password: string,
    firstName: string,
    lastName: string,
    phoneNumber: string,
    role: string,
  ) => Promise<boolean>;

  setSocialLogin: (token: string) => void;

  logout: () => void;
  clearError: () => void;

  // Dashboard actions
  fetchDashboard: () => Promise<boolean>;
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      dashboard: null,
      isDashboardLoading: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await authService.login(email, password);
          if (response.success) {
            set({
              isAuthenticated: true,
              token: response.data?.access_token || null,
              user: {
                email,
                ...response.data?.user,
              },
              isLoading: false,
            });
            return true;
          } else {
            set({
              error: response.error || "Login failed",
              isLoading: false,
            });
            return false;
          }
        } catch (error) {
          set({
            error: "Login failed",
            isLoading: false,
          });
          return false;
        }
      },

      signup: async (
        email: string,
        password: string,
        firstName: string,
        lastName: string,
        phoneNumber: string,
        role: string,
      ) => {
        set({ isLoading: true, error: null });
        try {
          const response = await authService.signup(
            email,
            password,
            firstName,
            lastName,
            phoneNumber,
            role,
            {
              email,
              password,
              first_name: firstName,
              last_name: lastName,
              phone_number: phoneNumber,
              role,
            },
          );

          if (response.success) {
            set({
              isAuthenticated: true,
              token: response.data?.access_token || null,
              user: {
                email,
                first_name: firstName,
                last_name: lastName,
                phone_number: phoneNumber,
                role: role,
                ...response.data?.user,
              },
              isLoading: false,
            });
            return true;
          } else {
            set({
              error: response.error || "Signup failed",
              isLoading: false,
            });
            return false;
          }
        } catch (error) {
          set({
            error: "Signup failed",
            isLoading: false,
          });
          return false;
        }
      },

      setSocialLogin: (token: string) => {
        set({
          token,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        });

        get().fetchDashboard();
      },

      logout: () => {
        authService.logout();
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          dashboard: null,
          error: null,
        });
      },

      clearError: () => {
        set({ error: null });
      },

      fetchDashboard: async () => {
        set({ isDashboardLoading: true });
        try {
          const response = await usersService.getDashboard();
          if (response.success && response.data) {
            set({
              dashboard: response.data,
              user: {
                ...get().user,
                email: response.data.user?.email || get().user?.email || "",
                first_name: response.data.first_name,
                last_name: response.data.last_name,
                role: response.data.user?.role || get().user?.role,
                coins: response.data.coins, // Synchronizes coins across app
              },
              isDashboardLoading: false,
            });
            return true;
          } else {
            set({
              error: response.error || "Failed to fetch dashboard",
              isDashboardLoading: false,
            });
            return false;
          }
        } catch (error) {
          set({
            error: "Failed to fetch dashboard",
            isDashboardLoading: false,
          });
          return false;
        }
      },
    }),
    {
      name: "auth-store",
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    },
  ),
);
