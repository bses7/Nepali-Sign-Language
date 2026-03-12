"use client";

import { useRef, useState, useEffect, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Mesh } from "three";
import * as THREE from "three";
import { OrbitControls, Environment } from "@react-three/drei";

interface AvatarMeshProps {
  avatarColor?: string;
}

interface Avatar3DViewerProps {
  avatarFolder?: string;
  className?: string;
}

const AvatarMesh: React.FC<AvatarMeshProps> = ({ avatarColor = "#4c1d95" }) => {
  const headRef = useRef<Mesh>(null);
  const bodyRef = useRef<Mesh>(null);

  useFrame(() => {
    if (headRef.current) headRef.current.rotation.y += 0.003;
    if (bodyRef.current) bodyRef.current.rotation.y += 0.003;
  });

  return (
    <group>
      <mesh ref={headRef} position={[0, 1.5, 0]}>
        <sphereGeometry args={[0.6, 32, 32]} />
        <meshStandardMaterial
          color={avatarColor}
          metalness={0.3}
          roughness={0.7}
          emissive={avatarColor}
          emissiveIntensity={0.2}
        />
      </mesh>
      <mesh position={[-0.2, 1.7, 0.5]}>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>
      <mesh position={[0.2, 1.7, 0.5]}>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>
      <mesh position={[-0.2, 1.7, 0.58]}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshStandardMaterial color="#000000" />
      </mesh>
      <mesh position={[0.2, 1.7, 0.58]}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshStandardMaterial color="#000000" />
      </mesh>
      <mesh ref={bodyRef} position={[0, 0.5, 0]}>
        <boxGeometry args={[0.6, 1.2, 0.4]} />
        <meshStandardMaterial
          color={avatarColor}
          metalness={0.2}
          roughness={0.8}
        />
      </mesh>
      {/* Arms & Legs */}
      <mesh position={[-0.5, 0.8, 0]}>
        <boxGeometry args={[0.3, 0.8, 0.3]} />
        <meshStandardMaterial color={avatarColor} />
      </mesh>
      <mesh position={[0.5, 0.8, 0]}>
        <boxGeometry args={[0.3, 0.8, 0.3]} />
        <meshStandardMaterial color={avatarColor} />
      </mesh>
      <mesh position={[-0.2, -0.8, 0]}>
        <boxGeometry args={[0.25, 0.8, 0.3]} />
        <meshStandardMaterial color={avatarColor} />
      </mesh>
      <mesh position={[0.2, -0.8, 0]}>
        <boxGeometry args={[0.25, 0.8, 0.3]} />
        <meshStandardMaterial color={avatarColor} />
      </mesh>
    </group>
  );
};

const AvatarScene: React.FC<{ avatarColor?: string }> = ({ avatarColor }) => (
  <>
    <ambientLight intensity={1} />
    <pointLight position={[10, 10, 10]} intensity={1.5} color="#fff" />
    <spotLight
      position={[0, 5, 0]}
      angle={0.3}
      penumbra={1}
      intensity={2}
      color={avatarColor}
    />
    <group position={[0, -0.5, 0]}>
      <AvatarMesh avatarColor={avatarColor} />
      <mesh position={[0, -1.2, 0]}>
        <cylinderGeometry args={[1.2, 1.5, 0.4, 32]} />
        <meshStandardMaterial color="#1a1a1a" metalness={0.8} roughness={0.2} />
      </mesh>
    </group>
    <Environment preset="night" />
    <OrbitControls
      autoRotate
      autoRotateSpeed={4}
      enableZoom={false}
      minPolarAngle={Math.PI / 3}
      maxPolarAngle={Math.PI / 2}
    />
  </>
);

export const Avatar3DViewer: React.FC<Avatar3DViewerProps> = ({
  avatarFolder = "default",
  className = "",
}) => {
  const [mounted, setMounted] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null); // 1. Added container ref

  useEffect(() => setMounted(true), []);

  const getAvatarColor = (folder: string) => {
    const colorMap: Record<string, string> = {
      indigo: "#4c1d95",
      purple: "#7c3aed",
      blue: "#1e40af",
      cyan: "#0891b2",
      gold: "#b45309",
      emerald: "#047857",
      default: "#5F7A61",
    };
    return colorMap[folder] || colorMap["default"];
  };

  const avatarColor = getAvatarColor(avatarFolder);

  if (!mounted)
    return (
      <div
        className={`w-full h-full bg-slate-900 animate-pulse rounded-2xl ${className}`}
      />
    );

  return (
    <div
      ref={containerRef} // 2. Attach ref to the wrapper div
      className={`w-full h-full rounded-2xl overflow-hidden border-4 border-primary/30 shadow-2xl ${className}`}
      style={{ position: "relative" }}
    >
      <Canvas
        key={avatarFolder}
        eventSource={containerRef as any} // 3. Explicitly set event source
        eventPrefix="client"
        camera={{ position: [0, 1, 3.5], fov: 45 }}
        gl={{ antialias: true, alpha: false }}
      >
        <Suspense fallback={null}>
          {" "}
          {/* 4. Added Suspense for React 19 stability */}
          <AvatarScene avatarColor={avatarColor} />
        </Suspense>
      </Canvas>
    </div>
  );
};

export default Avatar3DViewer;
