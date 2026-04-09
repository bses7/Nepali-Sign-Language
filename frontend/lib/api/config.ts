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
      badgesall: "/api/v1/users/badges/all",
      claim_challenge: "/api/v1/users/claim-challenge",
      getNotifications: "/api/v1/users/notifications",
      markNotification: (id: number) =>
        `/api/v1/users/notifications/${id}/read`,
    },
    lessons: {
      signs: "/api/v1/lessons/signs",
      sign_id: (id: string) => `/api/v1/lessons/signs/${id}`,
      complete_sign: "/api/v1/lessons/complete",
      upload_sign: "/api/v1/lessons/upload-sign",
    },
    avatar: {
      store: "/api/v1/avatars/store",
      equip: (avatarId: number) => `/api/v1/avatars/equip/${avatarId}`,
      buy: (avatarId: number) => `/api/v1/avatars/purchase/${avatarId}`,
    },
    practice: {
      submit: "/api/v1/practice/save-results",
    },
    quiz: {
      generate: (category: string, difficulty: string) =>
        `/api/v1/quiz/generate?category=${category}&difficulty=${difficulty}`,
      submit: "/api/v1/quiz/submit",
    },
    teachers: {
      submit_sign: "/api/v1/teacher/upload-sign",
    },
    admin: {
      getTeachersList: "/api/v1/admin/teachers",
      verifyTeacher: (userId: number) => `/api/v1/admin/verify-teacher/${userId}`,
    }
  },
};

export const getFullURL = (endpoint: string): string => {
  return `${API_CONFIG.baseURL}${endpoint}`;
};
