import Hero3DScene from "@/components/hero-3d-scene";
import { GameButton } from "@/components/game-button";
import { ThemeToggle } from "@/components/theme-toggle";
import Link from "next/link";

export default function Home() {
  return (
    <main className="w-full min-h-screen">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-sm bg-background/95 border-b border-border">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-foreground">SignLearn</h1>
          <div className="flex gap-3 items-center">
            <ThemeToggle />
            <Link href="/login">
              <GameButton variant="glossy" size="md">
                Sign In
              </GameButton>
            </Link>
            <Link href="/signup">
              <GameButton variant="retro" size="md">
                Get Started
              </GameButton>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section with 3D Scene */}
      <section className="relative w-full min-h-screen pt-20 pb-20 flex items-center justify-center overflow-hidden">
        <div className="max-w-7xl w-full mx-auto px-4 md:px-6">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <div className="space-y-8 z-10">
              <div className="space-y-4">
                <h1 className="text-5xl md:text-6xl font-bold text-foreground text-balance">
                  Master Sign Language, Learn at Your Pace
                </h1>
                <p className="text-xl text-muted-foreground text-balance max-w-lg">
                  An immersive gamified platform to learn Nepali Sign Language
                  with 3D visualizations, achievements, and a vibrant community.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4 pt-4">
                <Link href="/signup">
                  <GameButton variant="retro" size="md">
                    Start Learning
                  </GameButton>
                </Link>
                <Link href="#features">
                  <GameButton variant="duolingo" size="md">
                    Learn More
                  </GameButton>
                </Link>
              </div>

              {/* Stats */}
              <div className="flex gap-8 pt-8 border-t border-border">
                <div>
                  <div className="text-3xl font-bold text-primary">10K+</div>
                  <p className="text-sm text-muted-foreground">
                    Active Learners
                  </p>
                </div>
                <div>
                  <div className="text-3xl font-bold text-accent">500+</div>
                  <p className="text-sm text-muted-foreground">Sign Lessons</p>
                </div>
                <div>
                  <div className="text-3xl font-bold text-secondary">98%</div>
                  <p className="text-sm text-muted-foreground">
                    Completion Rate
                  </p>
                </div>
              </div>
            </div>

            {/* Right 3D Scene */}
            <div className="hidden md:block">
              <Hero3DScene />
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section
        id="features"
        className="py-20 px-4 md:px-6 bg-card/50 border-t border-b border-border"
      >
        <div className="max-w-6xl mx-auto">
          <div className="text-center space-y-4 mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground text-balance">
              Why SignLearn?
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Everything you need to master sign language effectively
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="p-8 rounded-xl border border-border bg-background hover:bg-card hover:border-primary/50 transition-all duration-300">
              <div className="text-3xl mb-4">🎮</div>
              <h3 className="text-xl font-bold text-foreground mb-3">
                Gamified Learning
              </h3>
              <p className="text-muted-foreground">
                Earn XP, build streaks, unlock achievements, and compete on
                leaderboards as you progress.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="p-8 rounded-xl border border-border bg-background hover:bg-card hover:border-primary/50 transition-all duration-300">
              <div className="text-3xl mb-4">🎯</div>
              <h3 className="text-xl font-bold text-foreground mb-3">
                3D Learning
              </h3>
              <p className="text-muted-foreground">
                Interactive 3D demonstrations and realistic animations for
                authentic sign language instruction.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="p-8 rounded-xl border border-border bg-background hover:bg-card hover:border-primary/50 transition-all duration-300">
              <div className="text-3xl mb-4">📊</div>
              <h3 className="text-xl font-bold text-foreground mb-3">
                Track Progress
              </h3>
              <p className="text-muted-foreground">
                Detailed statistics and personalized recommendations to keep you
                on the right path.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 md:px-6">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <div className="space-y-4">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground text-balance">
              Ready to Start Your Journey?
            </h2>
            <p className="text-lg text-muted-foreground">
              Join thousands of learners already mastering sign language
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Link href="/signup">
              <GameButton variant="primary" size="lg">
                Start Free Today
              </GameButton>
            </Link>
            <Link href="/login">
              <GameButton variant="duolingo" size="lg">
                Sign In
              </GameButton>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-12 px-4 md:px-6">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8 mb-12">
            <div>
              <h3 className="font-bold text-lg mb-4 text-foreground">
                SignLearn
              </h3>
              <p className="text-muted-foreground">
                Master sign language through gamified learning.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4 text-foreground">Product</h4>
              <ul className="space-y-2 text-muted-foreground">
                <li>
                  <a href="#" className="hover:text-foreground transition">
                    Features
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition">
                    Lessons
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition">
                    Pricing
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4 text-foreground">Company</h4>
              <ul className="space-y-2 text-muted-foreground">
                <li>
                  <a href="#" className="hover:text-foreground transition">
                    About
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition">
                    Contact
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition">
                    Blog
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4 text-foreground">Legal</h4>
              <ul className="space-y-2 text-muted-foreground">
                <li>
                  <a href="#" className="hover:text-foreground transition">
                    Privacy
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition">
                    Terms
                  </a>
                </li>
              </ul>
            </div>
          </div>
          <div className="border-t border-border pt-8 text-center text-muted-foreground">
            <p>&copy; 2026 SignLearn. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </main>
  );
}
