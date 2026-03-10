import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class", ".dark"],
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        display: ["var(--font-poppins)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "monospace"],
      },
      colors: {
        background: "#F4EDE4", // Warm Cream
        foreground: "#2C3E33", // Dark Sage for text
        card: "#E6E2D3",
        "card-foreground": "#2C3E33",
        primary: {
          DEFAULT: "#5F7A61", // Sage Green
          foreground: "#F4EDE4",
        },
        secondary: {
          DEFAULT: "#A3B9A0",
          foreground: "#2C3E33",
        },
        accent: {
          DEFAULT: "#8CBF9F",
          foreground: "#F4EDE4",
        },
        cta: {
          DEFAULT: "#5F7A61",
          foreground: "#F4EDE4",
        },
        destructive: {
          DEFAULT: "#C94C4C",
          foreground: "#F4EDE4",
        },
        muted: {
          DEFAULT: "#BFC9B4",
          foreground: "#2C3E33",
        },
        border: "#DAD5C9",
        input: "#E6E2D3",
        ring: "#5F7A61",
        success: "#6B8E55",
        warning: "#D9A95C",
        info: "#5F7A61",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
        slideIn: {
          from: { opacity: "0", transform: "translateY(10px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        pulse: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: ".5" },
        },
        bounce: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        glow: {
          "0%, 100%": { boxShadow: "0 0 5px rgba(95, 122, 97, 0.5)" },
          "50%": { boxShadow: "0 0 20px rgba(95, 122, 97, 0.8)" },
        },
        wiggle: {
          "0%, 100%": { transform: "rotate(-3deg)" },
          "50%": { transform: "rotate(3deg)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        "sparkle-spin": {
          "0%": { transform: "rotate(0deg) scale(1)" },
          "50%": { transform: "rotate(180deg) scale(1.2)" },
          "100%": { transform: "rotate(360deg) scale(1)" },
        },
        shine: {
          "0%": {
            transform: "translateX(-100%) translateY(-100%) rotate(45deg)",
          },
          "100%": {
            transform: "translateX(100%) translateY(100%) rotate(45deg)",
          },
        },
      },
      animation: {
        shimmer: "shimmer 2s linear infinite",
        slideIn: "slideIn 0.3s ease-out",
        pulse: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        bounce: "bounce 1s infinite",
        glow: "glow 2s ease-in-out infinite",
        wiggle: "wiggle 2s ease-in-out infinite",
        float: "float 3s ease-in-out infinite",
        "sparkle-pulse": "sparkle-spin 4s linear infinite",
        shine: "shine 5s infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
