"use client";

import { useRef, useState, useEffect, Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import {
  OrbitControls,
  Environment,
  useGLTF,
  useAnimations,
  ContactShadows,
  PerspectiveCamera,
  useProgress, // Added to track loading progress
} from "@react-three/drei";
import { avatarService } from "@/lib/api/avatar";
import { Loader2, Zap } from "lucide-react";

interface Avatar3DViewerProps {
  avatarFolder?: string;
  className?: string;
  animationName?: string;
  cameraPosition?: [number, number, number];
  cameraFov?: number;
  stagePosition?: [number, number, number];
  shadowPosition?: [number, number, number];
}

function HologramLoader() {
  const { progress } = useProgress();

  return (
    <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-slate-950/80 backdrop-blur-md">
      <div className="relative flex items-center justify-center">
        <div className="absolute w-32 h-32 border-4 border-primary/20 border-t-primary rounded-full animate-spin" />
        <div className="absolute w-24 h-24 border-4 border-blue-500/10 border-b-blue-500 rounded-full animate-spin [animation-duration:1.5s]" />

        <div className="flex flex-col items-center">
          <span className="text-white font-black text-xl italic leading-none">
            {Math.round(progress)}%
          </span>
          <span className="text-primary font-black text-[8px] uppercase tracking-widest mt-1">
            Loaded
          </span>
        </div>
      </div>

      <div className="mt-8 text-center space-y-2">
        <p className="text-white font-black text-xs uppercase tracking-[0.3em] animate-pulse">
          Materializing Unit
        </p>
        <div className="flex gap-1 justify-center">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="w-1 h-1 bg-primary rounded-full animate-bounce"
              style={{ animationDelay: `${i * 0.1}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

type GLTFResult = THREE.Object3D & {
  scene: THREE.Group;
  animations: THREE.AnimationClip[];
};

function LoadedAvatar({ folder, animation = "Idle", position }: any) {
  const group = useRef<THREE.Group>(null);
  const modelPath = avatarService.getAnimationUrl(folder, animation);
  const { scene, animations } = useGLTF(modelPath) as unknown as GLTFResult;
  const { actions } = useAnimations(animations, group);

  const isDragging = useRef(false);
  const previousX = useRef(0);

  useEffect(() => {
    const firstAction = Object.values(actions)[0] as THREE.AnimationAction;
    if (firstAction) {
      firstAction.reset().fadeIn(0.5).play();
    }
    return () => {
      if (firstAction) firstAction.fadeOut(0.5);
    };
  }, [actions, animation, folder]);

  // 👉 Mouse events
  useEffect(() => {
    const handleDown = (e: MouseEvent) => {
      isDragging.current = true;
      previousX.current = e.clientX;
    };

    const handleUp = () => {
      isDragging.current = false;
    };

    const handleMove = (e: MouseEvent) => {
      if (!isDragging.current || !group.current) return;

      const delta = e.clientX - previousX.current;
      previousX.current = e.clientX;

      group.current.rotation.y += delta * 0.01; // adjust sensitivity
    };

    window.addEventListener("mousedown", handleDown);
    window.addEventListener("mouseup", handleUp);
    window.addEventListener("mousemove", handleMove);

    return () => {
      window.removeEventListener("mousedown", handleDown);
      window.removeEventListener("mouseup", handleUp);
      window.removeEventListener("mousemove", handleMove);
    };
  }, []);

  return (
    <primitive
      ref={group}
      object={scene}
      scale={2.5}
      position={position}
      dispose={null}
    />
  );
}

function DojoStage({ position }: { position: [number, number, number] }) {
  return (
    <group position={position}>
      <mesh receiveShadow>
        <cylinderGeometry args={[1.5, 1.7, 0.2, 32]} />
        <meshStandardMaterial color="#2C3E33" metalness={0.8} />
      </mesh>
      <mesh position={[0, -0.05, 0]}>
        <cylinderGeometry args={[1.8, 1.9, 0.1, 32]} />
        <meshStandardMaterial
          color="#5F7A61"
          emissive="#5F7A61"
          emissiveIntensity={0.2}
        />
      </mesh>
    </group>
  );
}

export const Avatar3DViewer: React.FC<Avatar3DViewerProps> = ({
  avatarFolder = "avatar",
  className = "",
  animationName = "Idle",
  cameraPosition = [1, 0.5, 5.5],
  cameraFov = 45,
  stagePosition = [0, -3.2, -1],
  shadowPosition = [0, -3.2, -1],
}) => {
  const [mounted, setMounted] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted)
    return (
      <div
        className={`w-full h-full bg-slate-900 animate-pulse ${className}`}
      />
    );

  return (
    <div
      ref={containerRef}
      className={`w-full h-full bg-slate-950 ${className}`}
      style={{ position: "relative" }}
    >
      <Canvas
        shadows
        dpr={[1, 2]}
        gl={{ antialias: true }}
        eventSource={containerRef as any}
        eventPrefix="client"
      >
        <PerspectiveCamera
          makeDefault
          position={cameraPosition}
          fov={cameraFov}
        />

        <Suspense fallback={null}>
          <ambientLight intensity={0.8} />
          <spotLight
            position={[5, 10, 5]}
            angle={0.3}
            penumbra={1}
            intensity={2.5}
            castShadow
          />
          <pointLight position={[-5, 2, 5]} intensity={1} color="#5F7A61" />

          <DojoStage position={stagePosition} />
          <LoadedAvatar
            folder={avatarFolder}
            animation={animationName}
            position={[
              stagePosition[0],
              stagePosition[1] + 0.1,
              stagePosition[2],
            ]}
          />

          <ContactShadows
            position={shadowPosition}
            opacity={0.6}
            scale={10}
            blur={2.5}
            far={4}
          />
          <Environment preset="city" />

          <OrbitControls
            enableZoom={false}
            enablePan={false}
            enableRotate={false}
            target={[0, stagePosition[1] + 2.5, 0]}
            minPolarAngle={Math.PI / 3}
            maxPolarAngle={Math.PI / 1.8}
          />
        </Suspense>
      </Canvas>

      <Suspense fallback={null}>
        <ProgressWatcher />
      </Suspense>
    </div>
  );
};

function ProgressWatcher() {
  const { active } = useProgress();
  return active ? <HologramLoader /> : null;
}

export default Avatar3DViewer;
