import React from "react";

export function GameShopIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 512 512"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Blue Shop Base */}
      <path
        d="M106 312V426C106 442.569 119.431 456 136 456H236V312H106Z"
        fill="#3B82F6"
      />
      <path
        d="M276 456H376C392.569 456 406 442.569 406 426V312H276V456Z"
        fill="#3B82F6"
      />
      <path
        d="M406 312H106C106 250 140 220 180 220H332C372 220 406 250 406 312Z"
        fill="#2563EB"
      />

      {/* Red Controller Top */}
      <path
        d="M160 80H352C429.32 80 492 142.68 492 220C492 297.32 429.32 360 352 360H160C82.68 360 20 297.32 20 220C20 142.68 82.68 80 160 80Z"
        fill="#EF4444"
      />

      {/* Controller Buttons (White) */}
      <rect x="110" y="210" width="60" height="20" rx="4" fill="white" />
      <rect x="130" y="190" width="20" height="60" rx="4" fill="white" />

      <circle cx="360" cy="180" r="15" fill="white" />
      <circle cx="400" cy="220" r="15" fill="white" />
      <circle cx="360" cy="260" r="15" fill="white" />
      <circle cx="320" cy="220" r="15" fill="white" />
    </svg>
  );
}
