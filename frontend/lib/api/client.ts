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

      const isFormData =
        typeof FormData !== "undefined" && options.body instanceof FormData;

      const headers: Record<string, string> = {
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

      if (response.status === 401) {
        tokenManager.removeToken();
        if (typeof window !== "undefined") {
          window.location.href = "/login";
        }
        return { success: false, error: "Session expired. Please login." };
      }

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
    body?: Record<string, any> | URLSearchParams | FormData,
    options?: RequestOptions,
  ) {
    return this.request<T>(endpoint, {
      ...options,
      method: "POST",
      body:
        body instanceof FormData || body instanceof URLSearchParams
          ? body
          : JSON.stringify(body),
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
