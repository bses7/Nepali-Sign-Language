import { apiClient } from "./client";
import { API_CONFIG } from "./config";

export const adminService = {
  async getTeachersList() {
    return await apiClient.get<any[]>(
      API_CONFIG.endpoints.admin.getTeachersList,
    );
  },

  async verifyTeacher(userId: number) {
    return await apiClient.post(
      API_CONFIG.endpoints.admin.verifyTeacher(userId),
      {},
    );
  },
};
