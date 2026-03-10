'use client'

import { Badge } from '@/components/game-stats'
import { GameButton } from '@/components/game-button'
import Link from 'next/link'

export default function AchievementsPage() {
  const achievements = [
    { title: 'First Steps', description: 'Complete your first lesson', icon: '👣', isUnlocked: true, rarity: 'common', unlockedDate: '2026-02-15' },
    { title: 'Week Warrior', description: 'Maintain a 7-day streak', icon: '⚔️', isUnlocked: true, rarity: 'rare', unlockedDate: '2026-02-20' },
    { title: 'Month Master', description: 'Reach level 10', icon: '👑', isUnlocked: true, rarity: 'epic', unlockedDate: '2026-03-01' },
    { title: 'Perfect Day', description: 'Complete all daily challenges for 7 days', icon: '⭐', isUnlocked: false, rarity: 'legendary' },
    { title: 'Century Club', description: 'Earn 10,000 XP', icon: '💯', isUnlocked: false, rarity: 'legendary' },
    { title: 'Speed Demon', description: 'Complete 5 lessons in one day', icon: '⚡', isUnlocked: false, rarity: 'epic' },
    { title: 'Alphabet Expert', description: 'Complete all alphabet lessons', icon: '🔤', isUnlocked: true, rarity: 'rare', unlockedDate: '2026-02-25' },
    { title: 'Social Butterfly', description: 'Share 5 lessons with friends', icon: '🦋', isUnlocked: false, rarity: 'rare' },
    { title: 'Quiz Master', description: 'Get 100% on 10 quizzes', icon: '🎯', isUnlocked: false, rarity: 'epic' },
    { title: 'Community Helper', description: 'Help 5 learners with questions', icon: '🤝', isUnlocked: false, rarity: 'rare' },
    { title: 'Collector', description: 'Unlock 5 badge skins in the shop', icon: '🎨', isUnlocked: false, rarity: 'rare' },
    { title: 'Unstoppable', description: 'Reach a 100-day streak', icon: '🔥', isUnlocked: false, rarity: 'legendary' },
  ]

  const unlockedCount = achievements.filter(a => a.isUnlocked).length
  const totalCount = achievements.length

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-background via-card to-background">
      <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md bg-background/80 border-b border-border">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/dashboard" className="font-display text-2xl font-bold bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent hover:scale-105 transition-transform">
            SignLearn
          </Link>
          <Link href="/dashboard">
            <GameButton variant="accent" size="sm">Back</GameButton>
          </Link>
        </div>
      </nav>

      <main className="pt-24 pb-12">
        <section className="px-4 py-8 max-w-6xl mx-auto space-y-8">
          <div className="space-y-4">
            <h1 className="font-display text-5xl font-black">Achievement <span className="text-primary">Hall of Fame</span></h1>
            <p className="text-lg text-muted-foreground">
              Unlock badges by completing challenges and milestones. You've unlocked {unlockedCount} out of {totalCount} achievements!
            </p>
          </div>

          {/* Progress Stats */}
          <div className="glass rounded-3xl p-8 border-2 border-accent/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-muted-foreground mb-2">Overall Progress</p>
                <h3 className="font-display text-4xl font-bold">{Math.round((unlockedCount / totalCount) * 100)}%</h3>
              </div>
              <div className="w-32 h-32 rounded-full border-4 border-accent relative flex items-center justify-center">
                <div
                  className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary border-r-secondary"
                  style={{
                    transform: `rotate(${(unlockedCount / totalCount) * 360}deg)`,
                  }}
                />
                <p className="font-display text-2xl font-bold">{unlockedCount}/{totalCount}</p>
              </div>
            </div>
          </div>

          {/* Achievements Grid */}
          <div className="space-y-6">
            <h2 className="font-display text-2xl font-bold">All Achievements</h2>
            
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {achievements.map((achievement, i) => (
                <div key={i} className="flex flex-col items-center gap-4">
                  <Badge
                    title={achievement.title}
                    icon={achievement.icon}
                    isUnlocked={achievement.isUnlocked}
                    rarity={achievement.rarity}
                  />
                  <div className="text-center w-full">
                    <p className={`text-sm font-semibold ${achievement.isUnlocked ? '' : 'text-muted-foreground'}`}>
                      {achievement.title}
                    </p>
                    <p className="text-xs text-muted-foreground mb-2">{achievement.description}</p>
                    {achievement.isUnlocked && achievement.unlockedDate && (
                      <p className="text-xs text-success font-semibold">Unlocked {achievement.unlockedDate}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Tips Section */}
          <div className="glass rounded-3xl p-8 border-2 border-warning/20 space-y-4">
            <h3 className="font-display text-xl font-bold">How to Unlock More Achievements</h3>
            <ul className="space-y-2 text-muted-foreground">
              <li>• Complete lessons consistently to build streaks</li>
              <li>• Score 100% on quizzes to unlock Quiz Master</li>
              <li>• Reach higher levels by earning XP</li>
              <li>• Share your progress with the community</li>
              <li>• Help other learners on the platform</li>
            </ul>
          </div>
        </section>
      </main>
    </div>
  )
}
