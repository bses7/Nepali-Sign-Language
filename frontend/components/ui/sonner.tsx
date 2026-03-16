"use client";

import { useTheme } from "next-themes";
import { Toaster as Sonner, ToasterProps } from "sonner";

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = "system" } = useTheme();

  return (
    <Sonner
      theme={theme as ToasterProps["theme"]}
      className="toaster group"
      position="top-right"
      richColors
      closeButton
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-white/70 dark:group-[.toaster]:bg-zinc-900/70 backdrop-blur-xl border border-white/20 dark:border-zinc-800 shadow-2xl rounded-xl",
          title: "text-sm font-semibold",
          description: "text-xs opacity-80",
          actionButton: "bg-black text-white dark:bg-white dark:text-black",
          cancelButton:
            "bg-zinc-200 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100",
        },
      }}
      style={
        {
          "--normal-bg": "transparent",
          "--normal-text": "var(--foreground)",
          "--normal-border": "transparent",
        } as React.CSSProperties
      }
      {...props}
    />
  );
};

export { Toaster };
