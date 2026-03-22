"use client";

import { Html, useGLTF } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

export function ClassroomEnv({ targetChar }: { targetChar: string }) {
  const { scene: wallArtScene } = useGLTF("/world/wallart.glb");
  const { scene: potScene } = useGLTF("/world/pot.glb");

  const artModel = useMemo(() => wallArtScene.clone(), [wallArtScene]);
  const potModel = useMemo(() => potScene.clone(), [potScene]);

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]} receiveShadow>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#e2e8f0" roughness={0.1} metalness={0.1} />
      </mesh>
      <gridHelper
        args={[100, 50, "#cbd5e1", "#f1f5f9"]}
        position={[0, -1.99, 0]}
      />

      <mesh position={[0, 8, -12]} receiveShadow>
        <planeGeometry args={[100, 25]} />
        <meshStandardMaterial color="#5F7A61" />
      </mesh>

      <group position={[0, 5, -11.9]}>
        <mesh>
          <boxGeometry args={[18, 10, 0.1]} />
          <meshStandardMaterial color="#ffffff" roughness={0.1} />
        </mesh>
        <mesh position={[0, 0, -0.05]}>
          <boxGeometry args={[18.8, 10.8, 0.1]} />
          <meshStandardMaterial color="#2C3E33" />
        </mesh>

        {/* LEFT SIDE: TARGET CHARACTER */}
        <Html transform position={[-5, 0, 0.1]} distanceFactor={10}>
          <div className="flex flex-col items-center select-none pointer-events-none w-[400px]">
            <span className="text-[12px] font-black uppercase text-slate-400 tracking-[0.5em] mb-4">
              Target Sign
            </span>
            <h2 className="text-[140px] font-black text-slate-900 leading-none drop-shadow-sm">
              {targetChar}
            </h2>
            <div className="mt-8 px-6 py-2 bg-primary text-white rounded-2xl border-b-4 border-green-800 shadow-lg">
              <span className="font-black uppercase text-[10px] tracking-widest">
                Practice Lesson
              </span>
            </div>
          </div>
        </Html>

        {/* RIGHT SIDE: PROTOCOL */}
        <Html transform position={[5, 0, 0.1]} distanceFactor={10}>
          <div className="w-[250px] backdrop-blur-sm p-6 rounded-[2rem] flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
              <span className="text-[10px] font-black uppercase text-primary tracking-widest">
                Training Protocol
              </span>
            </div>
            <ul className="space-y-3">
              <li className="text-xs font-bold text-slate-700 flex gap-2">
                <span className="text-primary font-black">1.</span> Position
                hand in camera view (You can sign using either hand).
              </li>
              <li className="text-xs font-bold text-slate-700 flex gap-2">
                <span className="text-primary font-black">2.</span> Match
                instructor's finger shapes and angles.
              </li>
              <li className="text-xs font-bold text-slate-700 flex gap-2">
                <span className="text-primary font-black">3.</span> Hold still
                for 3 seconds after the sync queue.
              </li>
            </ul>
          </div>
        </Html>
      </group>

      <primitive
        object={artModel}
        position={[-14, 6, -11.8]}
        scale={6.5}
        rotation={[0, Math.PI / 1, 0]}
      />

      <primitive
        object={potModel}
        position={[12, -2, -10]}
        scale={0.5}
        rotation={[0, Math.PI / 4, 0]}
      />

      <mesh position={[0, 15, -5]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[8, 8, 0.5, 32]} />
        <meshStandardMaterial
          color="white"
          emissive="white"
          emissiveIntensity={0.1}
          transparent
          opacity={0.1}
        />
      </mesh>
    </group>
  );
}

useGLTF.preload("/world/wallart.glb");
useGLTF.preload("/world/pot.glb");
