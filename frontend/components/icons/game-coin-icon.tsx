import React from "react";

export function GameCoinIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* 3D Edge */}
      <circle cx="50" cy="54" r="42" fill="#D97706" />
      {/* Main Face */}
      <circle cx="50" cy="50" r="42" fill="#FBBF24" />
      {/* Inner Ring */}
      <circle cx="50" cy="50" r="34" stroke="#F59E0B" strokeWidth="4" />

      {/* Star Symbol */}
      <path
        d="M50 28L55.5 41.5H70L58.5 50L63 64L50 55.5L37 64L41.5 50L30 41.5H44.5L50 28Z"
        fill="#D97706"
      />

      {/* Glossy Highlight Layer */}
      <path
        d="M25 25C35 15 65 15 75 25"
        stroke="white"
        strokeWidth="6"
        strokeLinecap="round"
        opacity="0.4"
      />
    </svg>
  );
}
