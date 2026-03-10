'use client'

import { GameButton } from '@/components/game-button'
import { LevelCircle } from '@/components/game-stats'
import Link from 'next/link'

interface LessonCard {
  id: number
  title: string
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced'
  progress: number
  xpReward: number
  isLocked: boolean
  icon: string
  description: string
}

const lessons: LessonCard[] = [
  {
    id: 1,
    title: 'Alphabet Basics',
    difficulty: 'Beginner',
    progress: 85,
    xpReward: 100,
    isLocked: false,
    icon: '🔤',
    description: 'Learn the fundamentals of Nepali Sign Language alphabet'
  },
  {
    id: 2,
    title: 'Numbers 1-10',
    difficulty: 'Beginner',
    progress: 60,
    xpReward: 100,
    isLocked: false,
    icon: '🔢',
    description: 'Master basic number signs and counting'
  },
  {
    id: 3,
    title: 'Common Phrases',
    difficulty: 'Beginner',
    progress: 0,
    xpReward: 150,
    isLocked: false,
    icon: '💬',
    description: 'Essential phrases for everyday communication'
  },
  {
    id: 4,
    title: 'Family & Relations',
    difficulty: 'Intermediate',
    progress: 0,
    xpReward: 200,
    isLocked: false,
    icon: '👨‍👩‍👧',
    description: 'Learn signs for family members and relationships'
  },
  {
    id: 5,
    title: 'Daily Activities',
    difficulty: 'Intermediate',
    progress: 0,
    xpReward: 200,
    isLocked: true,
    icon: '🏃',
    description: 'Signs for common daily routines and activities'
  },
  {
    id: 6,
    title: 'Complex Conversations',
    difficulty: 'Intermediate',
    progress: 0,
    xpReward: 250,
    isLocked: true,
    icon: '🗣️',
    description: 'Build fluency in more complex dialogues'
  },
  {
    id: 7,
    title: 'Professional Signs',
    difficulty: 'Advanced',
    progress: 0,
    xpReward: 300,
    isLocked: true,
    icon: '💼',
    description: 'Workplace and professional communication'
  },
  {
    id: 8,
    title: 'Advanced Expressions',
    difficulty: 'Advanced',
    progress: 0,
    xpReward: 350,
    isLocked: true,
    icon: '✨',
    description: 'Express complex emotions and abstract concepts'
  },
]

export default function LessonsPage() {
  const difficultyColors = {
    Beginner: 'border-success',
    Intermediate: 'border-warning',
    Advanced: 'border-destructive',
  }

  const difficultyBg = {
    Beginner: 'bg-success/10',
    Intermediate: 'bg-warning/10',
    Advanced: 'bg-destructive/10',
  }

  const difficultyVariant = {
    Beginner: 'success' as const,
    Intermediate: 'warning' as const,
    Advanced: 'accent' as const,
  }

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-background via-card to-background">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md bg-background/80 border-b border-border">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/dashboard" className="font-display text-2xl font-bold bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent hover:scale-105 transition-transform">
            SignLearn
          </Link>
          <Link href="/dashboard">
            <GameButton variant="accent" size="sm">
              Back to Dashboard
            </GameButton>
          </Link>
        </div>
      </nav>

      <main className="pt-24 pb-12">
        {/* Header */}
        <section className="px-4 py-8 max-w-7xl mx-auto text-center space-y-4">
          <h1 className="font-display text-5xl md:text-6xl font-black">
            Lesson <span className="text-primary">Roadmap</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Progress through structured lessons to master Nepali Sign Language. Each lesson includes 3D visualizations and interactive exercises.
          </p>
        </section>

        {/* Filter & Sort */}
        <section className="px-4 py-6 max-w-7xl mx-auto">
          <div className="flex flex-wrap gap-3 justify-center">
            {['All', 'Beginner', 'Intermediate', 'Advanced'].map((filter) => (
              <button
                key={filter}
                className={`px-6 py-2 rounded-full font-semibold transition-all ${
                  filter === 'All'
                    ? 'bg-primary text-primary-foreground'
                    : 'glass border-2 border-border hover:border-primary'
                }`}
              >
                {filter}
              </button>
            ))}
          </div>
        </section>

        {/* Lessons Grid */}
        <section className="px-4 py-8 max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {lessons.map((lesson) => (
              <Link
                key={lesson.id}
                href={lesson.isLocked ? '#' : `/lessons/${lesson.id}`}
              >
                <div
                  className={`glass rounded-3xl p-8 space-y-6 transition-all hover:scale-105 border-2 ${
                    lesson.isLocked
                      ? 'border-muted opacity-50 cursor-not-allowed'
                      : `${difficultyColors[lesson.difficulty]} hover:shadow-xl`
                  }`}
                >
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div className="text-5xl">{lesson.icon}</div>
                    {lesson.isLocked ? (
                      <div className="text-2xl">🔒</div>
                    ) : (
                      <div
                        className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${difficultyBg[lesson.difficulty]}`}
                      >
                        {lesson.difficulty}
                      </div>
                    )}
                  </div>

                  {/* Content */}
                  <div className="space-y-2">
                    <h3 className="font-display text-2xl font-bold">{lesson.title}</h3>
                    <p className="text-sm text-muted-foreground">{lesson.description}</p>
                  </div>

                  {/* Progress Bar */}
                  {!lesson.isLocked && lesson.progress > 0 && (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center text-xs">
                        <span className="font-semibold">Progress</span>
                        <span className="text-muted-foreground">{lesson.progress}%</span>
                      </div>
                      <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-300"
                          style={{ width: `${lesson.progress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Footer */}
                  <div className="flex items-center justify-between pt-4 border-t border-border">
                    <div className="flex items-center gap-2">
                      <span className="text-xl">⭐</span>
                      <span className="font-bold text-sm">{lesson.xpReward} XP</span>
                    </div>
                    <button
                      className={`px-4 py-2 rounded-full font-semibold text-xs transition-all ${
                        lesson.isLocked
                          ? 'bg-muted text-muted-foreground cursor-not-allowed'
                          : lesson.progress === 100
                          ? 'bg-success text-white hover:bg-success/90'
                          : 'bg-primary text-primary-foreground hover:bg-primary/90'
                      }`}
                      onClick={(e) => {
                        if (lesson.isLocked) e.preventDefault()
                      }}
                    >
                      {lesson.isLocked ? 'Locked' : lesson.progress === 100 ? 'Review' : 'Start'}
                    </button>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Progression Info */}
        <section className="px-4 py-12 max-w-7xl mx-auto">
          <div className="glass rounded-3xl p-8 border-2 border-accent/20 space-y-6">
            <h2 className="font-display text-2xl font-bold">How Lessons Work</h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="space-y-2">
                <div className="text-4xl">👀</div>
                <h3 className="font-semibold">Watch & Learn</h3>
                <p className="text-sm text-muted-foreground">
                  View 3D animated demonstrations of each sign with detailed explanations
                </p>
              </div>
              <div className="space-y-2">
                <div className="text-4xl">🎯</div>
                <h3 className="font-semibold">Practice & Interact</h3>
                <p className="text-sm text-muted-foreground">
                  Complete interactive exercises to reinforce your understanding
                </p>
              </div>
              <div className="space-y-2">
                <div className="text-4xl">✅</div>
                <h3 className="font-semibold">Test & Advance</h3>
                <p className="text-sm text-muted-foreground">
                  Pass quizzes to unlock new lessons and earn achievements
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
