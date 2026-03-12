import { apiClient } from "./client";
import { API_CONFIG } from "./config";

export const lessonsService = {
  async getSigns() {
    return await apiClient.get<any[]>(API_CONFIG.endpoints.lessons.signs);
  },
};
