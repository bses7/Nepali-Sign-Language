"use client";

import { useRef, useMemo, useState, useEffect, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  Float,
  Center,
  Environment,
  ContactShadows,
  PerspectiveCamera,
  RoundedBox,
} from "@react-three/drei";
import * as THREE from "three";

const generateCoins = () => {
  return Array.from({ length: 12 }).map(() => ({
    position: [
      (Math.random() - 0.5) * 1.1,
      0.8,
      (Math.random() - 0.5) * 0.8,
    ] as [number, number, number],
    rotation: [Math.PI / 2, 0, Math.random() * Math.PI] as [
      number,
      number,
      number,
    ],
  }));
};

function FrontEmblem() {
  return (
    <group position={[0, 0, 0.61]}>
      <mesh>
        <planeGeometry args={[0.5, 0.6]} />
        <meshStandardMaterial color="#1f1f23" />
      </mesh>
      <mesh position={[0, 0, 0.01]}>
        <boxGeometry args={[0.38, 0.48, 0.05]} />
        <meshStandardMaterial color="#f2bf26" metalness={0.4} roughness={0.3} />
      </mesh>
      <mesh position={[0, 0, 0.04]} rotation={[0, 0, Math.PI / 4]}>
        <boxGeometry args={[0.45, 0.1, 0.05]} />
        <meshStandardMaterial color="#1f1f23" />
      </mesh>
    </group>
  );
}

function TreasureChest({
  isOpen,
  onClick,
}: {
  isOpen: boolean;
  onClick: () => void;
}) {
  const lid = useRef<THREE.Group>(null);
  const coins = useMemo(() => generateCoins(), []);

  useFrame(() => {
    if (!lid.current) return;
    const target = isOpen ? -Math.PI / 1.5 : 0;
    lid.current.rotation.x = THREE.MathUtils.lerp(
      lid.current.rotation.x,
      target,
      0.1,
    );
  });

  const handleInteraction = (e: any) => {
    e.stopPropagation();
    if (!isOpen) {
      console.log("Chest clicked!");
      onClick();
    }
  };

  const metalMat = (
    <meshStandardMaterial color="#1A261E" metalness={0.8} roughness={0.8} />
  );
  const goldMat = (
    <meshStandardMaterial color="#D9A95C" metalness={1} roughness={0.1} />
  );
  const sageMat = <meshStandardMaterial color="#5F7A61" roughness={0.8} />;

  return (
    <group rotation={[0, -0.4, 0]}>
      <group position={[0, 0.35, 0]} onPointerDown={handleInteraction}>
        <RoundedBox
          args={[1.6, 0.8, 1.2]}
          radius={0.05}
          smoothness={4}
          castShadow
        >
          {sageMat}
        </RoundedBox>
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[1.62, 0.2, 0.8]} />
          {metalMat}
        </mesh>
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.6, 0.82, 1.22]} />
          {metalMat}
        </mesh>
        {[
          [-0.75, 0.3, 0.55],
          [0.75, 0.3, 0.55],
          [-0.75, -0.3, 0.55],
          [0.75, -0.3, 0.55],
        ].map((pos, i) => (
          <mesh key={i} position={pos as any}>
            <sphereGeometry args={[0.06, 16, 16]} />
            {goldMat}
          </mesh>
        ))}
        <mesh position={[0, 0.2, 0.61]}>
          <boxGeometry args={[0.3, 0.3, 0.05]} />
          <meshStandardMaterial color="#F4EDE4" />
        </mesh>
      </group>
      <group
        visible={isOpen || !!(lid.current && lid.current.rotation.x < -0.1)}
      >
        <mesh position={[0, 0.5, 0]}>
          <cylinderGeometry args={[0.7, 0.7, 0.2, 16]} />
          <meshStandardMaterial
            color="#D9A95C"
            emissive="#D9A95C"
            emissiveIntensity={0.5}
          />
        </mesh>
        {coins.map((coin, i) => (
          <mesh key={i} position={coin.position} rotation={coin.rotation}>
            <cylinderGeometry args={[0.1, 0.1, 0.03, 10]} />
            {goldMat}
          </mesh>
        ))}
      </group>
      <group
        ref={lid}
        position={[0, 0.75, -0.6]}
        onPointerDown={handleInteraction}
      >
        <group position={[0, 0.2, 0.6]}>
          <RoundedBox
            args={[1.7, 0.5, 1.3]}
            radius={0.2}
            smoothness={4}
            castShadow
          >
            {sageMat}
          </RoundedBox>
          <mesh position={[0, 0, 0]}>
            <boxGeometry args={[0.65, 0.35, 1.35]} />
            {metalMat}
          </mesh>
          <mesh position={[0, -0.2, 0]}>
            <boxGeometry args={[1.75, 0.1, 1.45]} />
            {sageMat}
          </mesh>
          <mesh position={[0, -0.25, 0.65]}>
            <boxGeometry args={[0.2, 0.4, 0.2]} />
            {goldMat}
          </mesh>
          {[
            [-0.2, 0.25, 0],
            [0.2, 0.25, 0],
          ].map((pos, i) => (
            <mesh key={i} position={pos as any}>
              <sphereGeometry args={[0.08, 16, 16]} />
              {goldMat}
            </mesh>
          ))}
        </group>
      </group>
      {isOpen && (
        <pointLight
          position={[0, 1, 0.5]}
          color="#D9A95C"
          intensity={15}
          distance={5}
        />
      )}
    </group>
  );
}

export function GameChest3D({
  isOpen,
  onOpen,
}: {
  isOpen: boolean;
  onOpen: () => void;
}) {
  const [mounted, setMounted] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="w-full h-full min-h-[500px] bg-slate-950/20 animate-pulse rounded-[2.5rem]" />
    );
  }

  return (
    <div
      ref={containerRef} // Attach ref to the actual DOM element
      style={{
        width: "100%",
        height: "100%",
        minHeight: "500px",
        position: "relative",
      }}
    >
      <Canvas
        shadows
        dpr={[1, 2]}
        // Force events to attach to this div specifically, fixing the 'null' error
        eventSource={containerRef as any}
        eventPrefix="client"
      >
        <Suspense fallback={null}>
          <PerspectiveCamera makeDefault position={[0, 0.3, 8]} fov={30} />
          <ambientLight intensity={0.7} />
          <spotLight
            position={[10, 15, 10]}
            angle={0.25}
            penumbra={1}
            intensity={2.5}
            castShadow
          />
          <pointLight position={[-5, 5, -5]} intensity={1} color="#F4EDE4" />
          <pointLight position={[0, -2, 5]} intensity={0.5} color="#D9A95C" />

          <Center top>
            <Float speed={2.5} rotationIntensity={0.4} floatIntensity={0.5}>
              <TreasureChest isOpen={isOpen} onClick={onOpen} />
            </Float>
          </Center>

          <ContactShadows
            position={[0, -0.01, 0]}
            opacity={0.5}
            scale={15}
            blur={2.5}
            far={10}
          />
          <Environment preset="city" />
        </Suspense>
      </Canvas>
    </div>
  );
}
