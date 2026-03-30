"use client";

import { useGLTF, Html } from "@react-three/drei";
import { useMemo, useEffect } from "react";
import * as THREE from "three";
import { BrainCircuit, Lock } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner"; // Added to show "Locked" message

interface QuizLampProps {
  isLocked: boolean;
  isCompleted: boolean;
  onClick: () => void;
}

export function QuizLamp({ isLocked, isCompleted, onClick }: QuizLampProps) {
  const { scene } = useGLTF("/world/lamp.glb");
  const clonedScene = useMemo(() => scene.clone(), [scene]);

  useEffect(() => {
    clonedScene.traverse((child: any) => {
      if (child.isMesh) {
        child.castShadow = true;
        if (
          child.name.toLowerCase().includes("light") ||
          child.name.toLowerCase().includes("glass")
        ) {
          child.material = child.material.clone();
          if (!isLocked) {
            child.material.emissive = new THREE.Color(
              isCompleted ? "#22c55e" : "#fbbf24",
            );
            child.material.emissiveIntensity = 2.5;
          } else {
            child.material.emissiveIntensity = 0;
            child.material.color = new THREE.Color("#1a1a1a");
          }
        }
      }
    });
  }, [clonedScene, isLocked, isCompleted]);

  const handleClick = (e: any) => {
    e.stopPropagation();
    if (isLocked) {
      // Show feedback if the user clicks a locked quiz
      toast.error("Quest Locked", {
        description: "Complete all lessons in this tier to unlock the exam!",
      });
      return;
    }
    onClick();
  };

  return (
    <group
      onClick={handleClick}
      onPointerOver={() => (document.body.style.cursor = "pointer")}
      onPointerOut={() => (document.body.style.cursor = "auto")}
    >
      {/* 1. THE TOP ICON BADGE */}
      <Html
        transform
        sprite
        distanceFactor={25}
        position={[0, 3.2, 0]}
        // FIX: Keeps this icon behind the preview card
        zIndexRange={[0, 5]}
      >
        <div
          className={cn(
            "p-1.5 rounded-2xl border-b-2 shadow-2xl transition-all duration-500",
            isLocked
              ? "bg-slate-800 border-slate-950"
              : isCompleted
                ? "bg-[#76c92e] border-[#4e901a]"
                : "bg-blue-600 border-blue-800 animate-pulse",
          )}
        >
          {isLocked ? (
            <Lock size={14} className="text-slate-400" />
          ) : (
            <BrainCircuit size={14} className="text-white" />
          )}
        </div>
      </Html>

      {/* 2. THE LAMP MODEL */}
      <primitive object={clonedScene} scale={2.5} position={[0, 0.8, 0]} />

      {/* 3. LIGHT SOURCE */}
      {!isLocked && (
        <pointLight
          position={[0, 3.8, 0]}
          color={isCompleted ? "#22c55e" : "#fbbf24"}
          intensity={8}
          distance={10}
          castShadow
        />
      )}
    </group>
  );
}

useGLTF.preload("/world/lamp.glb");
