import { apiClient } from "./client";
import { API_CONFIG } from "./config";

export const quizService = {
  async generateQuiz(category: string, difficulty: string) {
    return await apiClient.get<any>(
      API_CONFIG.endpoints.quiz.generate(category, difficulty),
    );
  },

  async submitQuiz(data: {
    score: number;
    category: string;
    difficulty: string;
  }) {
    return await apiClient.post(API_CONFIG.endpoints.quiz.submit, data);
  },
};
