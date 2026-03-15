"use client";

import React, { Suspense, useEffect, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  Environment,
  ContactShadows,
  useGLTF,
  useAnimations,
  PerspectiveCamera,
  Float,
} from "@react-three/drei";
import * as THREE from "three";
import { Play, Pause, RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";

type GLTFResult = THREE.Object3D & {
  scene: THREE.Group;
  animations: THREE.AnimationClip[];
};

function AvatarModel({
  url,
  animationName,
  speed,
  isPaused,
  resetKey,
  isDraggingProgress,
  progressRef, // New Ref prop
  progressBarRef, // New Ref prop
}: any) {
  const group = useRef<THREE.Group>(null);
  const fullUrl = url.startsWith("http") ? url : `http://localhost:8000${url}`;
  const { scene, animations } = useGLTF(fullUrl) as unknown as GLTFResult;
  const { actions, names } = useAnimations(animations, group);

  useEffect(() => {
    const action = actions[animationName] || actions[names[0]];
    if (action) {
      action.reset().fadeIn(0.5).play();
    }
    return () => {
      if (action) action.fadeOut(0.5);
    };
  }, [actions, animationName, names, resetKey]);

  useFrame(() => {
    const action = actions[animationName] || actions[names[0]];
    if (!action) return;
    const duration = action.getClip().duration;

    if (isDraggingProgress) {
      action.paused = true;
      // Sync animation to where the user dragged the slider
      action.time = progressRef.current * duration;
    } else {
      action.paused = isPaused;
      action.setEffectiveTimeScale(speed);

      // Update the progress ref and the UI bar directly (No Re-render)
      const currentProgress = action.time / duration;
      progressRef.current = currentProgress;

      if (progressBarRef.current) {
        progressBarRef.current.style.width = `${currentProgress * 100}%`;
      }

      if (currentProgress >= 0.99) action.time = 0;
    }
  });

  return (
    <primitive ref={group} object={scene} scale={2.2} position={[0, -0.9, 0]} />
  );
}

export function Sign3DViewer({
  modelUrl,
  animationName,
}: {
  modelUrl: string;
  animationName: string;
}) {
  const [speed, setSpeed] = useState(0.5);
  const [isPaused, setIsPaused] = useState(false);
  const [resetKey, setResetKey] = useState(0);
  const [isDraggingProgress, setIsDraggingProgress] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);

  // REFS FOR PERFORMANCE
  const progressRef = useRef(0);
  const progressBarRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault();
        setIsPaused((prev) => !prev);
      }
      if (e.code === "KeyR") {
        setResetKey((k) => k + 1);
        progressRef.current = 0;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleMove = (e: any) => {
    if (isDraggingProgress) return; // Stop rotation while scrubbing
    if (e.buttons !== 1 && e.type !== "touchmove") return;
    const movementX = e.movementX || 0;
    const movementY = e.movementY || 0;
    setRotation((prev) => ({
      y: prev.y + movementX * 0.01,
      x: Math.max(Math.min(prev.x + movementY * 0.01, 0.5), -0.5),
    }));
  };

  const handleWheel = (e: React.WheelEvent) => {
    setZoom((prev) => Math.min(Math.max(prev - e.deltaY * 0.001, 0.6), 2));
  };

  if (!mounted)
    return (
      <div className="w-full h-full bg-slate-950 animate-pulse rounded-[3rem]" />
    );

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMove}
      onWheel={handleWheel}
      className="relative w-full h-full bg-slate-950 rounded-[3rem] border-4 border-slate-800 shadow-2xl overflow-hidden group touch-none select-none"
    >
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-4 bg-black/60 backdrop-blur-sm p-6 rounded-[2.5rem] border border-white/10 shadow-2xl min-w-[320px]">
        {/* TIME BAR */}
        <div
          className="w-full px-2 py-1 mb-2"
          onMouseMove={(e) => e.stopPropagation()}
        >
          <div className="relative w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
            {/* Direct DOM manipulation via Ref for smoothness */}
            <div
              ref={progressBarRef}
              className="absolute top-0 left-0 h-full bg-primary rounded-full pointer-events-none"
              style={{ width: "0%" }}
            />
            <input
              type="range"
              min="0"
              max="1"
              step="0.001"
              defaultValue={0}
              onPointerDown={() => setIsDraggingProgress(true)}
              onPointerUp={() => setIsDraggingProgress(false)}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                progressRef.current = val;
                if (progressBarRef.current)
                  progressBarRef.current.style.width = `${val * 100}%`;
              }}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
            />
          </div>
        </div>

        {/* BUTTONS */}
        <div className="flex items-center gap-4">
          <button
            onClick={() => setIsPaused(!isPaused)}
            className="w-12 h-12 rounded-full bg-white flex items-center justify-center text-slate-900 active:scale-90 transition-all"
          >
            {isPaused ? (
              <Play fill="currentColor" size={20} />
            ) : (
              <Pause fill="currentColor" size={20} />
            )}
          </button>
          <button
            onClick={() => {
              setResetKey((k) => k + 1);
              progressRef.current = 0;
            }}
            className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center text-white active:scale-90 transition-all"
          >
            <RotateCcw size={18} />
          </button>
          <div className="w-px h-8 bg-white/20 mx-2" />
          <div className="flex gap-2">
            {[1, 0.5, 0.25].map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={cn(
                  "px-4 py-1.5 rounded-xl font-black text-[10px] uppercase border-b-4 transition-all active:translate-y-1 active:border-b-0",
                  speed === s
                    ? "bg-primary border-green-800 text-white"
                    : "bg-slate-800 text-slate-400",
                )}
              >
                {s === 1 ? "1.0x" : s === 0.5 ? "Slow" : "0.25x"}
              </button>
            ))}
          </div>
        </div>
      </div>

      <Canvas shadows dpr={[1, 2]}>
        <PerspectiveCamera makeDefault position={[0, 1.5, 10]} fov={25} />
        <Suspense fallback={null}>
          <ambientLight intensity={0.8} />
          <spotLight
            position={[5, 10, 5]}
            angle={0.3}
            penumbra={1}
            intensity={2}
            castShadow
          />
          <group position={[0, -1, 0]}>
            <mesh receiveShadow>
              <cylinderGeometry args={[2.5, 2.8, 0.2, 32]} />
              <meshStandardMaterial color="#2C3E33" metalness={0.6} />
            </mesh>
            <mesh position={[0, -0.05, 0]}>
              <cylinderGeometry args={[2.9, 3, 0.1, 32]} />
              <meshStandardMaterial
                color="#5F7A61"
                emissive="#5F7A61"
                emissiveIntensity={0.2}
              />
            </mesh>
          </group>
          <group rotation={[rotation.x, rotation.y, 0]} scale={zoom}>
            <Float
              speed={isPaused ? 0 : 2}
              rotationIntensity={0.1}
              floatIntensity={0.1}
            >
              <AvatarModel
                url={modelUrl}
                animationName={animationName}
                speed={speed}
                isPaused={isPaused}
                resetKey={resetKey}
                isDraggingProgress={isDraggingProgress}
                progressRef={progressRef}
                progressBarRef={progressBarRef}
              />
            </Float>
          </group>
          <ContactShadows
            position={[0, -1, 0]}
            opacity={0.4}
            scale={10}
            blur={2.5}
          />
          <Environment preset="city" />
        </Suspense>
      </Canvas>
    </div>
  );
}
