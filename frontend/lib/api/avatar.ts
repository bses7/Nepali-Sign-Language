import { apiClient } from "./client";
import { API_CONFIG } from "./config";

export interface AvatarItem {
  description: string;
  id: number;
  name: string;
  folder_name: string;
  price: number;
  is_owned: boolean;
  attributes: {
    type: string;
    gender: string;
    face_shape: string;
    skin_color: string;
    hair_color: string;
    eye_color: string;
    clothing_color: string;
    accessories: string[];
    shop_animations: string[]; 
  };
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
    return await apiClient.post(API_CONFIG.endpoints.avatar.buy(avatarId), {});
  },

  async equipAvatar(avatarId: number) {
    return await apiClient.post(
      API_CONFIG.endpoints.avatar.equip(avatarId),
      {},
    );
  },

  getThumbnailUrl(folder: string): string {
    return `${API_CONFIG.baseURL}/static/avatars/${folder}/thumbnail.png`;
  },

  getShopAnimationUrl(folder: string, animation: string): string {
    const fileName = encodeURIComponent(animation);
    return `${API_CONFIG.baseURL}/static/avatars/${folder}/shop/${fileName}.glb`;
  },
};
