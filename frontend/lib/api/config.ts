// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const API_CONFIG = {
  baseURL: API_BASE_URL,
  staticRoot: `${API_BASE_URL}/static`,
  version: "v1",
  endpoints: {
    auth: {
      login: "/api/v1/auth/login",
      signup: "/api/v1/users",
      passwordRecovery: "/api/v1/auth/password-recovery",
      passwordReset: "/api/v1/auth/reset-password",
    },
    users: {
      dashboard: "/api/v1/users/dashboard",
      claim: "/api/v1/users/claim-daily",
      leaderboard: "/api/v1/users/leaderboard",
      badges: "/api/v1/users/badges",
    },
    lessons: {
      signs: "/api/v1/lessons/signs",
      sign_id: (id: string) => `/api/v1/lessons/signs/${id}`,
      complete_sign: (id: string) => `/api/v1/lessons/signs/${id}/complete`,
    },
    avatar: {
      store: "/api/v1/avatars/store",
    }
  },
};

export const getFullURL = (endpoint: string): string => {
  return `${API_CONFIG.baseURL}${endpoint}`;
};
