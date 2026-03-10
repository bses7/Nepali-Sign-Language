'use client'

import { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Mesh } from 'three'
import * as THREE from 'three'
import { OrbitControls, Environment, Sphere, useGLTF } from '@react-three/drei'

const StaticGeometries = () => {
  const cubeRef = useRef<Mesh>(null)
  const pyramidRef = useRef<Mesh>(null)
  const sphereRef = useRef<Mesh>(null)

  useFrame((state) => {
    if (cubeRef.current) {
      cubeRef.current.rotation.x = 0.3
      cubeRef.current.rotation.y = 0.4
    }
    if (pyramidRef.current) {
      pyramidRef.current.rotation.z = 0.2
    }
    if (sphereRef.current) {
      sphereRef.current.rotation.x = 0.1
      sphereRef.current.rotation.y = 0.15
    }
  })

  return (
    <>
      <mesh ref={cubeRef} position={[0, 0, 0]}>
        <boxGeometry args={[2, 2, 2]} />
        <meshStandardMaterial
          color="#3b82f6"
          metalness={0.4}
          roughness={0.6}
        />
      </mesh>

      <mesh ref={pyramidRef} position={[-3.5, 0, 0]}>
        <coneGeometry args={[1.5, 3, 4]} />
        <meshStandardMaterial
          color="#1e40af"
          metalness={0.3}
          roughness={0.7}
        />
      </mesh>

      <mesh ref={sphereRef} position={[3.5, 0, 0]}>
        <icosahedronGeometry args={[1.2, 3]} />
        <meshStandardMaterial
          color="#0ea5e9"
          metalness={0.2}
          roughness={0.8}
        />
      </mesh>
    </>
  )
}

const Scene = () => {
  return (
    <>
      <color attach="background" args={['#f8f9fa']} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 10]} intensity={0.8} />
      <directionalLight position={[-5, 5, 5]} intensity={0.4} color="#60a5fa" />
      
      <StaticGeometries />
      
      <Environment preset="studio" />
      <OrbitControls autoRotate autoRotateSpeed={0.5} enableZoom={false} />
    </>
  )
}

export const Hero3DScene = () => {
  return (
    <div className="w-full h-[500px] md:h-[600px] relative rounded-2xl overflow-hidden bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <Canvas
        camera={{ position: [0, 0, 10], fov: 50 }}
        gl={{ antialias: true, alpha: false }}
      >
        <Scene />
      </Canvas>
    </div>
  )
}

export default Hero3DScene
