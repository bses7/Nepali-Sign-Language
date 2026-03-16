import type { Metadata } from "next";
import { Poppins, JetBrains_Mono, Plus_Jakarta_Sans } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { Gamepad2 } from "lucide-react";
import { Toaster } from "sonner";

const plusJakarta = Plus_Jakarta_Sans({
  weight: ["400", "500", "600", "700", "800"],
  subsets: ["latin"],
  variable: "--font-plus-jakarta",
});

const poppins = Poppins({
  weight: ["400", "500", "600", "700", "800"],
  subsets: ["latin"],
  variable: "--font-poppins",
});

const jetbrains = JetBrains_Mono({
  weight: ["400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

export const metadata: Metadata = {
  title: "SignLearn - Master Sign Language Through Gaming",
  description:
    "An immersive, gamified platform to learn Nepali Sign Language with 3D visualizations, achievements, and leaderboards.",
  generator: "v0.app",
  icons: {
    icon: [
      {
        url: "/icon-light-32x32.png",
        media: "(prefers-color-scheme: light)",
      },
      {
        url: "/icon-dark-32x32.png",
        media: "(prefers-color-scheme: dark)",
      },
      {
        url: "/icon.svg",
        type: "image/svg+xml",
      },
    ],
    apple: "/apple-icon.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="...">
        <SidebarProvider defaultOpen={false}>
          <AppSidebar />
          <main className="relative flex-1">
            {/* This button is the hamburger that appears globally */}
            <div className="fixed top-4 left-4 z-[60]">
              <SidebarTrigger className="group h-12 w-12 bg-white border-slate-200 transition-all hover:bg-slate-50 active:translate-y-1 active:border-b-0">
                <Gamepad2
                  size={24}
                  className="text-primary group-hover:rotate-12 group-hover:scale-150 transition-transform duration-200"
                />
              </SidebarTrigger>
            </div>
            {children}
            <Toaster
              position="top-center"
              toastOptions={{
                unstyled: true, 
                classNames: {
                  toast:
                    "w-full flex items-center gap-3 p-4 rounded-2xl border-b-4 shadow-2xl transition-all duration-300",
                  success: "bg-[#F4EDE4] border-primary text-primary",
                  error: "bg-[#FDF2F2] border-destructive text-destructive",
                  info: "bg-white border-blue-500 text-blue-600",
                  title: "font-black uppercase tracking-tight text-sm",
                  description: "font-bold text-xs opacity-80",
                },
              }}
            />
          </main>
        </SidebarProvider>
        <Analytics />
      </body>
    </html>
  );
}
