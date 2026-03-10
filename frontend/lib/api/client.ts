import { API_CONFIG, getFullURL } from "./config";

interface RequestOptions extends RequestInit {
  headers?: Record<string, string>;
}

interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
  success: boolean;
}

// Token management utilities
export const tokenManager = {
  getToken: (): string | null => {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("access_token");
  },
  setToken: (token: string): void => {
    if (typeof window === "undefined") return;
    localStorage.setItem("access_token", token);
  },
  removeToken: (): void => {
    if (typeof window === "undefined") return;
    localStorage.removeItem("access_token");
  },
};

// HTTP Client
export const apiClient = {
  async request<T>(
    endpoint: string,
    options: RequestOptions = {},
  ): Promise<ApiResponse<T>> {
    try {
      const url = getFullURL(endpoint);
      const token = tokenManager.getToken();

      // Check if we are sending Form Data
      const isFormData = options.body instanceof URLSearchParams;

      const headers: Record<string, string> = {
        // Only set default JSON header if it's NOT Form Data
        ...(isFormData ? {} : { "Content-Type": "application/json" }),
        ...options.headers,
      };

      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }

      const response = await fetch(url, {
        ...options,
        headers,
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error:
            data.detail?.[0]?.msg ||
            data.error ||
            data.message ||
            "An error occurred",
          data: undefined,
        };
      }

      return { success: true, data: data as T };
    } catch (error) {
      return {
        success: false,
        error: "Network error. Please check your connection.",
        data: undefined,
      };
    }
  },

  get<T>(endpoint: string, options?: RequestOptions) {
    return this.request<T>(endpoint, { ...options, method: "GET" });
  },

  post<T>(
    endpoint: string,
    body?: Record<string, any> | URLSearchParams,
    options?: RequestOptions,
  ) {
    return this.request<T>(endpoint, {
      ...options,
      method: "POST",
      body: body instanceof URLSearchParams ? body : JSON.stringify(body),
    });
  },

  put<T>(
    endpoint: string,
    body?: Record<string, any> | URLSearchParams,
    options?: RequestOptions,
  ) {
    return this.request<T>(endpoint, {
      ...options,
      method: "PUT",
      body: body instanceof URLSearchParams ? body : JSON.stringify(body),
    });
  },

  delete<T>(endpoint: string, options?: RequestOptions) {
    return this.request<T>(endpoint, { ...options, method: "DELETE" });
  },
};
