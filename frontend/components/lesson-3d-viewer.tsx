'use client'

import { Canvas, useFrame } from '@react-three/fiber'
import { useRef } from 'react'
import { Mesh } from 'three'
import { OrbitControls, Environment, Text3D } from '@react-three/drei'

const AnimatedSignCharacter = () => {
  const groupRef = useRef<any>(null)

  useFrame((state) => {
    if (!groupRef.current) return
    groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.2
  })

  return (
    <group ref={groupRef} position={[0, -1, 0]}>
      {/* Head */}
      <mesh position={[0, 2, 0]}>
        <sphereGeometry args={[0.4, 32, 32]} />
        <meshPhongMaterial color="#FDB4D6" />
      </mesh>

      {/* Body */}
      <mesh position={[0, 1, 0]}>
        <boxGeometry args={[0.6, 1, 0.3]} />
        <meshPhongMaterial color="#a78bfa" emissive="#7c3aed" emissiveIntensity={0.3} />
      </mesh>

      {/* Left Arm */}
      <mesh position={[-0.6, 1.2, 0]}>
        <boxGeometry args={[0.2, 1, 0.2]} />
        <meshPhongMaterial color="#fb7185" emissive="#f43f5e" emissiveIntensity={0.3} />
      </mesh>

      {/* Right Arm */}
      <mesh position={[0.6, 1.2, 0]}>
        <boxGeometry args={[0.2, 1, 0.2]} />
        <meshPhongMaterial color="#fb7185" emissive="#f43f5e" emissiveIntensity={0.3} />
      </mesh>
    </group>
  )
}

const LessonScene = () => {
  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, 10, 5]} intensity={0.8} color="#8b5cf6" />
      
      <AnimatedSignCharacter />
      
      <Environment preset="studio" />
      <OrbitControls autoRotate autoRotateSpeed={3} />
    </>
  )
}

export const Lesson3DViewer = () => {
  return (
    <div className="w-full h-96 rounded-2xl overflow-hidden border-2 border-primary/20">
      <Canvas
        camera={{ position: [0, 0, 4], fov: 75 }}
        gl={{ antialias: true, alpha: true }}
      >
        <LessonScene />
      </Canvas>
    </div>
  )
}

export default Lesson3DViewer
