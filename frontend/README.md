# SignLearn - Gamified Sign Language Learning Platform

A modern, immersive educational platform for learning Nepali Sign Language with game mechanics, 3D visualizations, and social features.

## Features

### 🎮 Gamification System
- **Experience Points (XP)**: Earn XP by completing lessons and challenges
- **Leveling System**: Progress through 25 levels with increasing difficulty
- **Daily Streaks**: Maintain consistency with streak tracking and rewards
- **Achievements/Badges**: Unlock 12+ achievements with rarity levels (Common, Rare, Epic, Legendary)
- **Leaderboards**: Global ranking system to compete with other learners
- **Coin Currency**: Earn coins to purchase avatar customizations

### 🌐 3D Visualizations
- **Interactive 3D Scenes**: Built with React Three Fiber (Three.js)
- **Hero Landing Page**: Animated geometric shapes demonstrating modern web aesthetics
- **Lesson 3D Viewer**: Interactive 3D character demonstrating sign language
- **Customizable Environments**: Studio and night-themed 3D backgrounds

### 🎨 Modern Design System
- **Vibrant Color Palette**: Purple (#a78bfa), Orange (#fb7185), Teal (#06b6d4), Green, Rose
- **Game-Style Buttons**: Duolingo-inspired rounded pill buttons with gradients and hover effects
- **Game Typography**: 
  - Headers: Press Start 2P font for retro gaming feel
  - Body: Poppins for clean readability
  - Code: JetBrains Mono for technical elements
- **Glass Morphism**: Glassmorphic cards with transparency effects
- **Responsive Design**: Mobile-first approach with Tailwind CSS

### 📚 Learning Structure
- **Progressive Lessons**: 8 lessons organized by difficulty (Beginner, Intermediate, Advanced)
- **Step-by-Step Learning**: Multi-step lessons with visual demonstrations
- **Interactive Quizzes**: Knowledge checks after each lesson
- **Unlocking System**: Later lessons require progress in earlier ones

### 🛍️ Avatar Shop
- **Avatar Skins**: 8 customizable avatars (Cyborg, Astronaut, Wizard, Ninja, etc.)
- **Badge Skins**: 4 achievement badge themes
- **Special Effects**: 3 XP effect animations
- **Coin Economy**: Earn coins through learning activities

### 📊 Dashboard
- **User Stats**: XP bar, current level, coin count
- **Streak Display**: Current and best streak tracking
- **Continue Learning**: Quick access to in-progress lessons
- **Daily Challenges**: Motivational daily tasks with progress tracking
- **Achievement Preview**: Recently unlocked badges

### 🏆 Social Features
- **Global Leaderboard**: See top 8 learners ranked by level and XP
- **Achievement Hall**: View all 12 achievements and unlock progress
- **Community Engagement**: Help other learners and share progress

## Pages

| Route | Purpose |
|-------|---------|
| `/` | Landing page with hero 3D scene |
| `/signup` | Account creation with age group selection |
| `/login` | User authentication |
| `/dashboard` | Main user hub with stats and quick actions |
| `/lessons` | Browse all lessons by difficulty |
| `/lessons/[id]` | Individual lesson with 3D viewer and quiz |
| `/leaderboard` | Global ranking system |
| `/achievements` | Achievement gallery and progress tracking |
| `/shop` | Avatar and cosmetic shop |

## Technology Stack

- **Framework**: Next.js 16 (App Router)
- **Frontend**: React 19, TypeScript
- **3D Graphics**: Three.js, React Three Fiber, @react-three/drei
- **Styling**: Tailwind CSS 4, CSS Variables (OKLCH color system)
- **UI Components**: Shadcn/ui with custom game-style variants
- **Fonts**: Press Start 2P, Poppins, JetBrains Mono

## Design Highlights

### Color System
- **Light Mode**: Clean whites with vibrant accent colors
- **Dark Mode**: Deep backgrounds with glowing accents
- **Brand Colors**: Purple (primary), Orange (secondary), Teal (accent), Green (success), Rose (warning)

### Button Styles
- Gradient fills with depth effects
- Rounded pill shapes for modern feel
- Hover animations and scale effects
- Active state feedback with shadow reduction
- Loading states with spinner animations
- Ripple effect on icon buttons

### Game Elements
- XP progress bars with gradient fills
- Animated streak display with emoji and pulse effects
- Level circles with rotating border animations
- Achievement badges with rarity colors
- Glowing effects on important elements

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Open http://localhost:3000
```

## Customization

### Colors
Edit `app/globals.css` to modify the color palette:
```css
:root {
  --primary: 270 85% 55%;      /* Purple */
  --secondary: 25 95% 55%;     /* Orange */
  --accent: 180 90% 45%;       /* Teal */
}
```

### Fonts
Adjust fonts in `app/layout.tsx`:
- `Press_Start_2P`: Game-style headers
- `Poppins`: Body text
- `JetBrains_Mono`: Code and technical content

## Future Enhancements

- Real-time multiplayer lessons
- Voice recognition for pronunciation
- Mobile app with gesture recognition
- User-generated content system
- Advanced analytics and learning insights
- Accessibility improvements (closed captions, audio descriptions)

---

Built with v0 for modern, engaging e-learning experiences.
