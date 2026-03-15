import { apiClient } from "./client";
import { API_CONFIG } from "./config";

export interface AvatarItem {
  description: string;
  id: number;
  name: string;
  folder_name: string;
  price: number;
  is_owned: boolean;
}

export const avatarService = {
  getAnimationUrl(folder: string, animation: string): string {
    const fileName = encodeURIComponent(animation);
    return `${API_CONFIG.baseURL}/static/avatars/${folder}/${fileName}.glb`;
  },

  async getAvatarStore() {
    return await apiClient.get<AvatarItem[]>(API_CONFIG.endpoints.avatar.store);
  },

  async buyAvatar(avatarId: number) {
    return await apiClient.post(`/avatars/${avatarId}/buy`, {});
  },

  async equipAvatar(avatarId: number) {
    return await apiClient.post(`/avatars/${avatarId}/equip`, {});
  },

  getThumbnailUrl(folder: string): string {
    return `${API_CONFIG.baseURL}/static/avatars/${folder}/thumbnail.png`;
  },
};
