import { apiClient } from "./client";
import { API_CONFIG } from "./config";

export const practiceService = {
  async saveResults(xpEarned: number) {
    try {
      const response = await apiClient.post(
        API_CONFIG.endpoints.practice.submit,
        { xpEarned },
      );
      return {
        success: true,
        data: response.data,
      };
    } catch (error: any) {
      console.error("Save Results Error:", error);
      return {
        success: false,
        error: error.response?.data?.detail || "Failed to save progress",
      };
    }
  },

  getWebSocketUrl(targetChar: string) {
    const baseUrl = API_CONFIG.baseURL || "http://localhost:8000/api/v1";
    const wsBase = baseUrl.replace(/^http/, "ws");

    return `${wsBase}/api/v1/practice/ws/${encodeURIComponent(targetChar)}`;
  },
};
