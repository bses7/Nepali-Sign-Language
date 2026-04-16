"use client";

import React, { Suspense, useRef, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  useGLTF,
  Float,
  PerspectiveCamera,
  ContactShadows,
  Environment,
  Html,
} from "@react-three/drei";
import { useRouter } from "next/navigation";
import { Sparkles, Wand2, BookMarked } from "lucide-react";
import * as THREE from "three";
import { cn } from "@/lib/utils";

// Path to your model: public/world/spellbook.glb
const MODEL_PATH = "/world/spellbook.glb";

function BookModel({ hovered }: { hovered: boolean }) {
  // Load the model
  const { scene } = useGLTF(MODEL_PATH);
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      const rotationSpeed = hovered ? 2.5 : 0.5;
      groupRef.current.rotation.y += 0.01 * rotationSpeed;

      groupRef.current.rotation.z = THREE.MathUtils.lerp(
        groupRef.current.rotation.z,
        hovered ? 0.2 : 0,
        0.1,
      );

      // 3. Smooth Scale Lerp
      const targetScale = hovered ? 1.2 : 0.9;
      groupRef.current.scale.lerp(
        new THREE.Vector3(targetScale, targetScale, targetScale),
        0.1,
      );
    }
  });

  return <primitive ref={groupRef} object={scene} dispose={null} />;
}

export function SpellbookFAB() {
  const router = useRouter();
  const [hovered, setHovered] = useState(false);

  return (
    <div className="fixed bottom-6 right-2 z-[100] group flex flex-col items-end">
      {/* Gamified Speech Bubble */}
      <div
        className={cn(
          "mb-2 transition-all duration-500 transform origin-bottom-right",
          hovered
            ? "opacity-100 scale-100 translate-y-0"
            : "opacity-0 scale-50 translate-y-10 pointer-events-none",
        )}
      >
        <div className="bg-white border-4 border-slate-900 p-4 rounded-3xl shadow-[8px_8px_0_0_#000] min-w-[220px] relative">
          <div className="flex items-center gap-2 mb-1">
            <Sparkles size={18} className="text-purple-500 animate-pulse" />
            <span className="font-black uppercase text-[10px] tracking-widest text-purple-600">
              NSL Lab
            </span>
          </div>
          <p className="font-black text-slate-900 text-sm uppercase leading-tight">
            Create custom signs <br />
            <span className="text-blue-600">with Magic AI</span>
          </p>
          {/* Decorative Corner */}
          <div className="absolute -top-2 -right-2 w-6 h-6 bg-yellow-400 border-2 border-slate-900 rotate-12 flex items-center justify-center shadow-md">
            <Wand2 size={12} className="text-slate-900" />
          </div>
        </div>
      </div>

      <button
        className="w-40 h-40 md:w-52 md:h-52 cursor-pointer relative outline-none transition-transform active:scale-90"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={() => router.push("/generation")}
      >
        <div
          className={cn(
            "absolute inset-0 bg-purple-500/20 rounded-full blur-[50px] transition-opacity duration-700",
            hovered ? "opacity-100 scale-125" : "opacity-0 scale-75",
          )}
        />

        <Canvas>
          <PerspectiveCamera makeDefault position={[0, 0.2, 5]} fov={40} />
          <ambientLight intensity={1.5} />
          <pointLight position={[10, 10, 10]} intensity={2} />
          <spotLight
            position={[-10, 10, 10]}
            angle={0.15}
            penumbra={1}
            intensity={2}
          />

          <Suspense fallback={null}>
            <Float speed={2.5} rotationIntensity={1} floatIntensity={2}>
              <BookModel hovered={hovered} />
            </Float>
            <ContactShadows
              position={[0, -1.5, 0]}
              opacity={0.4}
              scale={8}
              blur={2}
              far={4}
            />
            <Environment preset="city" />
          </Suspense>
        </Canvas>

        {!hovered && (
          <div className="absolute bottom-4 right-1/2 translate-x-1/2 animate-bounce bg-slate-900 text-white text-[8px] font-black px-2 py-1 rounded-full uppercase tracking-widest">
            Generate
          </div>
        )}
      </button>
    </div>
  );
}

// Preload the model to prevent flickering
useGLTF.preload(MODEL_PATH);
