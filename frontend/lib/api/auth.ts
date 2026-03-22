import { apiClient, tokenManager } from "./client";
import { API_CONFIG } from "./config";

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user?: {
    id: string;
    email: string;
    first_name?: string;
    last_name?: string;
  };
}

export interface SignupRequest {
  email: string;
  password: string;
  first_name: string;
  last_name: string;
  phone_number: string;
  role: string;
}

export interface SignupResponse {
  user: {
    id: string;
    email: string;
    first_name: string;
    last_name: string;
  };
  access_token: string;
}

export const authService = {
  async login(
    email: string,
    password: string,
  ): Promise<{ success: boolean; error?: string; data?: LoginResponse }> {
    const formData = new URLSearchParams();
    formData.append("username", email);
    formData.append("password", password);

    const response = await apiClient.post<LoginResponse>(
      API_CONFIG.endpoints.auth.login,
      formData,
      {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
      },
    );

    if (response.success && response.data?.access_token) {
      tokenManager.setToken(response.data.access_token);
      console.log("Login successful, token saved");
    }

    return response;
  },

  async signup(
    email: string,
    password: string,
    firstName: string,
    lastName: string,
    phoneNumber: string,
    role: string,
    signupData: SignupRequest,
  ): Promise<{ success: boolean; error?: string; data?: SignupResponse }> {
    const response = await apiClient.post<SignupResponse>(
      API_CONFIG.endpoints.auth.signup,
      signupData,
    );

    if (response.success && response.data?.access_token) {
      tokenManager.setToken(response.data.access_token);
      console.log("Signup successful, token saved");
    }

    return response;
  },

  logout(): void {
    tokenManager.removeToken();
    console.log("Logged out, token removed");
  },

  isAuthenticated(): boolean {
    return !!tokenManager.getToken();
  },

  async recoverPassword(
    email: string,
  ): Promise<{ success: boolean; error?: string }> {
    return await apiClient.post(API_CONFIG.endpoints.auth.passwordRecovery, {
      email,
    });
  },

  async resetPassword(
    password: string,
    token: string,
  ): Promise<{ success: boolean; error?: string }> {
    return await apiClient.post(
      API_CONFIG.endpoints.auth.passwordReset,
      {
        token: token,
        new_password: password,
      },
    );
  },
};
