import { apiClient } from "./client";
import { API_CONFIG } from "./config";

export const teachersService = {
  async uploadSign(formData: FormData) {
    return await apiClient.post(
      API_CONFIG.endpoints.teachers.submit_sign,
      formData
    );
  },
};
