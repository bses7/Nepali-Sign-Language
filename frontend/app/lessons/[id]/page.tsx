'use client'

import { GameButton } from '@/components/game-button'
import { XPBar } from '@/components/game-stats'
import Lesson3DViewer from '@/components/lesson-3d-viewer'
import Link from 'next/link'
import { useState } from 'react'

interface LessonParams {
  params: {
    id: string
  }
}

export default function LessonPage({ params }: LessonParams) {
  const lessonId = parseInt(params.id)
  const [currentStep, setCurrentStep] = useState(0)
  const [isQuizMode, setIsQuizMode] = useState(false)

  const lesson = {
    id: lessonId,
    title: 'Alphabet Basics',
    difficulty: 'Beginner',
    totalSteps: 5,
    xpReward: 100,
    description: 'Learn the fundamentals of Nepali Sign Language alphabet',
    steps: [
      {
        sign: 'A',
        description: 'Form a closed fist with your thumb on the side',
        tips: ['Keep your hand at shoulder level', 'Your palm should face forward'],
      },
      {
        sign: 'B',
        description: 'Open your hand with fingers together and thumb folded',
        tips: ['All fingers should point upward', 'Keep the palm facing forward'],
      },
      {
        sign: 'C',
        description: 'Form a C-shape with your thumb and fingers',
        tips: ['The curve should be smooth', 'Position at mouth level'],
      },
      {
        sign: 'D',
        description: 'Make an O-shape with your thumb and fingers at the mouth',
        tips: ['Connect thumb to index finger', 'Other fingers point upward'],
      },
      {
        sign: 'E',
        description: 'Form a fist and place it on your chin',
        tips: ['All fingers should be closed', 'Keep consistent pressure'],
      },
    ],
    quiz: [
      {
        question: 'Which sign requires forming a C-shape?',
        options: ['A', 'B', 'C', 'D'],
        correctAnswer: 2,
      },
      {
        question: 'At what level should you position sign B?',
        options: ['Chest', 'Mouth', 'Shoulder', 'Head'],
        correctAnswer: 2,
      },
    ],
  }

  const currentLessonStep = lesson.steps[currentStep]
  const isLastStep = currentStep === lesson.steps.length - 1

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-background via-card to-background">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md bg-background/80 border-b border-border">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/lessons" className="font-display text-2xl font-bold bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent hover:scale-105 transition-transform">
            SignLearn
          </Link>
          <div className="flex items-center gap-4">
            <div className="text-sm font-semibold text-muted-foreground">
              Step {currentStep + 1} / {lesson.totalSteps}
            </div>
            <Link href="/lessons">
              <GameButton variant="accent" size="sm">
                Back
              </GameButton>
            </Link>
          </div>
        </div>
      </nav>

      <main className="pt-24 pb-12">
        {/* Header */}
        <section className="px-4 py-8 max-w-7xl mx-auto space-y-4">
          <h1 className="font-display text-5xl font-black">
            {lesson.title} <span className="text-primary">- Part {currentStep + 1}</span>
          </h1>
          <p className="text-lg text-muted-foreground">
            Master each sign step by step with interactive 3D visualizations
          </p>
        </section>

        {!isQuizMode ? (
          <>
            {/* 3D Viewer Section */}
            <section className="px-4 py-8 max-w-7xl mx-auto">
              <div className="grid lg:grid-cols-2 gap-8">
                {/* 3D Viewer */}
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold">Watch & Learn</h2>
                  <Lesson3DViewer />
                  <p className="text-sm text-muted-foreground">
                    Rotate the model with your mouse to see the sign from different angles. Click and drag to explore.
                  </p>
                </div>

                {/* Sign Information */}
                <div className="space-y-6">
                  <div className="glass rounded-3xl p-8 border-2 border-primary/20 space-y-6">
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">Current Sign</p>
                      <h2 className="font-display text-6xl font-black text-primary">
                        {currentLessonStep.sign}
                      </h2>
                    </div>

                    <div className="space-y-2">
                      <h3 className="font-bold">How to Sign</h3>
                      <p className="text-muted-foreground">{currentLessonStep.description}</p>
                    </div>

                    <div className="space-y-3">
                      <h3 className="font-bold">💡 Tips for Success</h3>
                      {currentLessonStep.tips.map((tip, i) => (
                        <div key={i} className="flex gap-3 p-3 bg-success/10 rounded-lg border border-success/20">
                          <span className="text-lg">✓</span>
                          <p className="text-sm">{tip}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Progress */}
                  <div className="glass rounded-3xl p-8 border-2 border-accent/20 space-y-4">
                    <h3 className="font-bold">Lesson Progress</h3>
                    <XPBar
                      current={(currentStep / lesson.totalSteps) * 100}
                      max={100}
                      level={lesson.id}
                      showLabel={false}
                    />
                    <p className="text-sm text-muted-foreground">
                      Complete all steps to earn {lesson.xpReward} XP
                    </p>
                  </div>
                </div>
              </div>
            </section>

            {/* Navigation Buttons */}
            <section className="px-4 py-8 max-w-7xl mx-auto">
              <div className="flex gap-4 justify-between">
                <GameButton
                  variant="secondary"
                  size="lg"
                  disabled={currentStep === 0}
                  onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                >
                  Previous Sign
                </GameButton>

                {isLastStep ? (
                  <GameButton
                    variant="primary"
                    size="lg"
                    onClick={() => setIsQuizMode(true)}
                  >
                    Take Quiz
                  </GameButton>
                ) : (
                  <GameButton
                    variant="primary"
                    size="lg"
                    onClick={() => setCurrentStep(currentStep + 1)}
                  >
                    Next Sign
                  </GameButton>
                )}
              </div>
            </section>
          </>
        ) : (
          <>
            {/* Quiz Mode */}
            <section className="px-4 py-8 max-w-2xl mx-auto">
              <div className="glass rounded-3xl p-8 border-2 border-warning/20 space-y-8">
                <div>
                  <h2 className="font-display text-3xl font-bold mb-2">Knowledge Check</h2>
                  <p className="text-muted-foreground">
                    Test your understanding before moving forward
                  </p>
                </div>

                <div className="space-y-6">
                  {lesson.quiz.map((q, idx) => (
                    <div key={idx} className="space-y-4">
                      <p className="font-semibold text-lg">{idx + 1}. {q.question}</p>
                      <div className="grid gap-3">
                        {q.options.map((option, optIdx) => (
                          <button
                            key={optIdx}
                            className="p-4 rounded-2xl border-2 border-border hover:border-primary bg-card hover:bg-primary/10 transition-all text-left font-semibold"
                          >
                            {option}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex gap-4">
                  <GameButton
                    variant="secondary"
                    size="lg"
                    onClick={() => setIsQuizMode(false)}
                    className="flex-1"
                  >
                    Back to Lesson
                  </GameButton>
                  <GameButton
                    variant="primary"
                    size="lg"
                    className="flex-1"
                  >
                    Submit Answers
                  </GameButton>
                </div>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  )
}
