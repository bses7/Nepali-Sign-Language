// lib/api/generation.ts
import { apiClient } from "./client";
import { API_CONFIG } from "./config";

export interface SignGenerationResponse {
  text: string;
  avatar_used: string;
  model_url: string;
}

export interface SkeletonGenerationResponse {
  text: string;
  video_url: string;
}

export const generationService = {
  async generateSign(text: string) {
    return await apiClient.post<SignGenerationResponse>(
      API_CONFIG.endpoints.generation.generateSign,
      { text },
    );
  },

  async generateSkeleton(text: string) {
    return await apiClient.post<SkeletonGenerationResponse>(
      API_CONFIG.endpoints.generation.generateSkeleton,
      { text },
    );
  },
};
