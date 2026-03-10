"use client";

import { GameButton } from "@/components/game-button";
import { CoinDisplay } from "@/components/game-stats";
import Link from "next/link";
import { useState } from "react";

export default function ShopPage() {
  const [selectedTab, setSelectedTab] = useState<
    "avatars" | "badges" | "effects"
  >("avatars");
  const userCoins = 2540;

  const avatarSkins = [
    {
      id: 1,
      name: "Classic",
      price: 0,
      icon: "👤",
      owned: true,
      isDefault: true,
    },
    { id: 2, name: "Cyborg", price: 500, icon: "🤖", owned: true },
    { id: 3, name: "Astronaut", price: 800, icon: "🧑‍🚀", owned: false },
    { id: 4, name: "Wizard", price: 1200, icon: "🧙", owned: false },
    { id: 5, name: "Ninja", price: 1500, icon: "🥷", owned: false },
    { id: 6, name: "Dragon", price: 2000, icon: "🐉", owned: false },
    { id: 7, name: "Angel", price: 1800, icon: "😇", owned: false },
    { id: 8, name: "Phoenix", price: 2500, icon: "🔥", owned: false },
  ];

  const badgeSkins = [
    {
      id: 1,
      name: "Golden",
      price: 300,
      icon: "⭐",
      owned: true,
      isDefault: true,
    },
    { id: 2, name: "Neon", price: 500, icon: "✨", owned: false },
    { id: 3, name: "Holographic", price: 800, icon: "🌈", owned: false },
    { id: 4, name: "Fire", price: 600, icon: "🔥", owned: false },
  ];

  const effects = [
    {
      id: 1,
      name: "XP Burst",
      price: 200,
      icon: "💥",
      owned: true,
      isDefault: true,
    },
    { id: 2, name: "Rainbow Trail", price: 400, icon: "🌈", owned: false },
    { id: 3, name: "Star Shower", price: 600, icon: "⭐", owned: false },
  ];

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-background via-card to-background">
      <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md bg-background/80 border-b border-border">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link
            href="/dashboard"
            className="font-display text-2xl font-bold bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent hover:scale-105 transition-transform"
          >
            SignLearn
          </Link>
          <div className="flex items-center gap-4">
            <CoinDisplay amount={userCoins} />
            <Link href="/dashboard">
              <GameButton variant="accent" size="sm">
                Back
              </GameButton>
            </Link>
          </div>
        </div>
      </nav>

      <main className="pt-24 pb-12">
        <section className="px-4 py-8 max-w-7xl mx-auto space-y-8">
          <div className="space-y-4">
            <h1 className="font-display text-5xl font-black">
              Avatar <span className="text-secondary">Shop</span>
            </h1>
            <p className="text-lg text-muted-foreground">
              Customize your profile with unique skins and effects. Earn coins
              by completing lessons and maintaining streaks.
            </p>
          </div>

          {/* Tabs */}
          <div className="flex gap-3 flex-wrap">
            {(["avatars", "badges", "effects"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setSelectedTab(tab)}
                className={`px-6 py-3 rounded-full font-semibold transition-all ${
                  selectedTab === tab
                    ? "bg-primary text-primary-foreground"
                    : "glass border-2 border-border hover:border-primary"
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {/* Content */}
          {selectedTab === "avatars" && (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {avatarSkins.map((skin) => (
                <div
                  key={skin.id}
                  className={`glass rounded-2xl p-6 space-y-4 border-2 transition-all text-center ${
                    skin.isDefault
                      ? "border-primary bg-primary/10"
                      : skin.owned
                        ? "border-success"
                        : "border-border"
                  }`}
                >
                  <div className="text-6xl">{skin.icon}</div>
                  <div>
                    <p className="font-semibold">{skin.name}</p>
                    {!skin.owned && (
                      <p className="text-lg font-bold text-warning">
                        {skin.price} 💰
                      </p>
                    )}
                  </div>
                  <GameButton
                    variant={
                      skin.isDefault
                        ? "accent"
                        : skin.owned
                          ? "secondary"
                          : "primary"
                    }
                    size="sm"
                    className="w-full"
                    disabled={skin.price > userCoins && !skin.owned}
                  >
                    {skin.isDefault ? "Equipped" : skin.owned ? "Equip" : "Buy"}
                  </GameButton>
                </div>
              ))}
            </div>
          )}

          {selectedTab === "badges" && (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {badgeSkins.map((skin) => (
                <div
                  key={skin.id}
                  className={`glass rounded-2xl p-6 space-y-4 border-2 transition-all text-center ${
                    skin.isDefault
                      ? "border-primary bg-primary/10"
                      : skin.owned
                        ? "border-success"
                        : "border-border"
                  }`}
                >
                  <div className="text-6xl">{skin.icon}</div>
                  <div>
                    <p className="font-semibold">{skin.name}</p>
                    {!skin.owned && (
                      <p className="text-lg font-bold text-warning">
                        {skin.price} 💰
                      </p>
                    )}
                  </div>
                  <GameButton
                    variant={
                      skin.isDefault
                        ? "accent"
                        : skin.owned
                          ? "secondary"
                          : "primary"
                    }
                    size="sm"
                    className="w-full"
                    disabled={skin.price > userCoins && !skin.owned}
                  >
                    {skin.isDefault ? "Equipped" : skin.owned ? "Equip" : "Buy"}
                  </GameButton>
                </div>
              ))}
            </div>
          )}

          {selectedTab === "effects" && (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {effects.map((effect) => (
                <div
                  key={effect.id}
                  className={`glass rounded-2xl p-6 space-y-4 border-2 transition-all text-center ${
                    effect.isDefault
                      ? "border-primary bg-primary/10"
                      : effect.owned
                        ? "border-success"
                        : "border-border"
                  }`}
                >
                  <div className="text-6xl animate-bounce">{effect.icon}</div>
                  <div>
                    <p className="font-semibold">{effect.name}</p>
                    {!effect.owned && (
                      <p className="text-lg font-bold text-warning">
                        {effect.price} 💰
                      </p>
                    )}
                  </div>
                  <GameButton
                    variant={
                      effect.isDefault
                        ? "accent"
                        : effect.owned
                          ? "secondary"
                          : "primary"
                    }
                    size="sm"
                    className="w-full"
                    disabled={effect.price > userCoins && !effect.owned}
                  >
                    {effect.isDefault
                      ? "Active"
                      : effect.owned
                        ? "Activate"
                        : "Buy"}
                  </GameButton>
                </div>
              ))}
            </div>
          )}

          {/* How to Earn Info */}
          <div className="glass rounded-3xl p-8 border-2 border-accent/20 space-y-4">
            <h3 className="font-display text-xl font-bold">
              How to Earn Coins
            </h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div>
                <p className="text-2xl mb-2">📚</p>
                <p className="text-sm">
                  Complete lessons to earn 50-100 coins each
                </p>
              </div>
              <div>
                <p className="text-2xl mb-2">🔥</p>
                <p className="text-sm">
                  Maintain streaks for bonus coins daily
                </p>
              </div>
              <div>
                <p className="text-2xl mb-2">⭐</p>
                <p className="text-sm">Unlock achievements for extra rewards</p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
