import { create } from "zustand";
import { persist } from "zustand/middleware";
import { authService } from "@/lib/api/auth";
import { usersService, DashboardData } from "@/lib/api/users";

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
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  dashboard: DashboardData | null;
  isDashboardLoading: boolean;

  isOffline: boolean;

  login: (email: string, password: string) => Promise<boolean>;

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

  fetchDashboard: () => Promise<boolean>;
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      isOffline: false,
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
            if (response.error?.toLowerCase().includes("network error")) {
              set({ isOffline: true });
            }
            set({ error: response.error, isLoading: false });
            return false;
          }
        } catch (error) {
          set({ isOffline: true, isLoading: false });
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
            if (response.error?.toLowerCase().includes("network error")) {
              set({ isOffline: true });
            }
            set({ error: response.error, isLoading: false });
            return false;
          }
        } catch (error) {
          set({ isOffline: true, isLoading: false });
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
                phone_number: response.data.phone_number || "",
                role: response.data.user?.role || get().user?.role,
                coins: response.data.coins, 
              },
              isDashboardLoading: false,
              isOffline: false,
            });
            return true;
          } else {
            const errorStr = response.error?.toLowerCase() || "";
            if (
              errorStr.includes("network error") ||
              errorStr.includes("connection")
            ) {
              set({ isOffline: true });
            }

            set({ isDashboardLoading: false });
            return false;
          }
        } catch (error) {
          set({ isOffline: true, isDashboardLoading: false });
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
