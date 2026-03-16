import { apiClient } from "./client";
import { API_CONFIG } from "./config";

export const lessonsService = {
  async getSigns() {
    return await apiClient.get<any[]>(API_CONFIG.endpoints.lessons.signs);
  },

  async getSignById(id: string) {
    return await apiClient.get<any>(API_CONFIG.endpoints.lessons.sign_id(id));
  },

  async completeSign(signId: number) {
    return await apiClient.post(API_CONFIG.endpoints.lessons.complete_sign, {
      sign_id: signId,
    });
  },
};
