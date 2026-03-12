import { apiClient } from "./client";
import { API_CONFIG } from "./config";

export interface DashboardData {
  user: any;
  first_name: string;
  last_name: string;
  xp: number;
  level: number;
  streak_count: number;
  total_signs: number;
  completed_signs: number;
  progress_percentage: number;
  equipped_avatar_id: string;
  equipped_avatar_folder: string;
  coins: number;
  can_claim_daily: boolean;
}

export const usersService = {
  async getDashboard(): Promise<{
    success: boolean;
    error?: string;
    data?: DashboardData;
  }> {
    const response = await apiClient.get<DashboardData>(
      API_CONFIG.endpoints.users.dashboard,
    );

    if (response.success) {
      console.log("[v0] Dashboard data fetched successfully");
    } else {
      console.error("[v0] Failed to fetch dashboard:", response.error);
    }

    return response;
  },

  async getLeaderboard(): Promise<{
    success: boolean;
    data?: any[];
    error?: string;
  }> {
    return await apiClient.get(API_CONFIG.endpoints.users.leaderboard);
  },

  async getUserBadges() {
    return await apiClient.get<any[]>(API_CONFIG.endpoints.users.badges);
  },

  async claimDailyReward() {
    return await apiClient.post(API_CONFIG.endpoints.users.claim);
  },
};
