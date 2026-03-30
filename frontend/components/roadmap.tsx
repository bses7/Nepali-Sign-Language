"use client";

import React, { useMemo, useRef, useState, useEffect, Suspense } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  Environment,
  ContactShadows,
  PerspectiveCamera,
  Html,
  useGLTF,
  BakeShadows,
} from "@react-three/drei";
import * as THREE from "three";
import { MapPin } from "lucide-react";
import { RoadmapNode } from "./roadmap-node";
import { SignPreviewCard } from "./sign-preview-card";
import { QuizLamp } from "./quiz-node";
import { QuizPreviewCard } from "./quiz-preview-card";

// --- GLTF LOADING HELPER ---
function GltfModel({ path, ...props }: { path: string } & any) {
  const gltf = useGLTF(path) as any;
  const scene = gltf.scene;
  const clonedScene = useMemo(() => scene.clone(), [scene]);

  useEffect(() => {
    clonedScene.traverse((child: any) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        if (child.material) child.material.roughness = 1;
      }
    });
  }, [clonedScene]);

  return <primitive object={clonedScene} {...props} />;
}

// --- ENVIRONMENTAL COMPONENTS ---
const Tree = (props: any) => {
  const variant = useMemo(() => Math.floor(Math.random() * 4), []);
  const models = [
    "/world/Tree.glb",
    "/world/Tree1.glb",
    "/world/Tree2.glb",
    "/world/Tree3.glb",
  ];
  return <GltfModel path={models[variant]} {...props} />;
};

const Rock = (props: any) => {
  const variant = useMemo(() => Math.floor(Math.random() * 4), []);
  const models = [
    "/world/Rock.glb",
    "/world/Rock1.glb",
    "/world/Rock2.glb",
    "/world/Rock3.glb",
  ];
  return <GltfModel path={models[variant]} {...props} />;
};

const Grass = (props: any) => {
  const variant = useMemo(() => Math.floor(Math.random() * 2), []);
  const models = ["/world/grass.glb", "/world/grass2.glb"];
  return <GltfModel path={models[variant]} {...props} />;
};

// --- DRAG CONTROLLER ---
function WorldDragger({ totalLength, targetZ }: any) {
  const { gl, camera } = useThree();
  const isDragging = useRef(false);
  const lastY = useRef(0);
  const currentZ = useRef(0);
  const targetFOV = useRef(50);

  useEffect(() => {
    const canvas = gl.domElement;

    // 1. Zoom Logic (Wheel)
    const onWheel = (e: WheelEvent) => {
      targetFOV.current = Math.max(
        30,
        Math.min(60, targetFOV.current + e.deltaY * 0.05),
      );
    };

    // 2. Navigation Logic (Mouse/Touch Drag)
    const onDown = (e: any) => {
      isDragging.current = true;
      lastY.current = e.clientY || (e.touches ? e.touches[0].clientY : 0);
    };

    const onMove = (e: any) => {
      if (!isDragging.current) return;
      const y = e.clientY || (e.touches ? e.touches[0].clientY : 0);
      const delta = (lastY.current - y) * 0.12;
      targetZ.current = Math.max(
        -totalLength,
        Math.min(10, targetZ.current - delta),
      );
      lastY.current = y;
    };

    const onUp = () => (isDragging.current = false);

    // 3. --- NEW: Keyboard Navigation Logic ---
    const onKeyDown = (e: KeyboardEvent) => {
      const moveStep = 4; // How many units to move per keypress

      if (e.key === "ArrowUp" || e.key === "w" || e.key === "W") {
        targetZ.current = Math.max(-totalLength, targetZ.current - moveStep);
      }
      if (e.key === "ArrowDown" || e.key === "s" || e.key === "S") {
        targetZ.current = Math.min(10, targetZ.current + moveStep);
      }
    };

    // Event Listeners
    canvas.addEventListener("wheel", onWheel, { passive: true });
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("touchstart", onDown);
    window.addEventListener("touchmove", onMove, { passive: false });
    window.addEventListener("touchend", onUp);

    // Add Keyboard listener to window
    window.addEventListener("keydown", onKeyDown);

    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("touchstart", onDown);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [gl, totalLength, targetZ]);

  useFrame(() => {
    const pCamera = camera as THREE.PerspectiveCamera;
    if (pCamera.fov !== targetFOV.current) {
      pCamera.fov = THREE.MathUtils.lerp(pCamera.fov, targetFOV.current, 0.1);
      pCamera.updateProjectionMatrix();
    }
    // Smooth Lerp for the actual movement
    currentZ.current = THREE.MathUtils.lerp(
      currentZ.current,
      targetZ.current,
      0.08,
    );
    camera.position.z = 18 + currentZ.current;
    camera.lookAt(0, 0, currentZ.current - 5);
  });
  return null;
}

// --- MAIN ROADMAP ---
export function Roadmap3D({
  signs,
  avatarFolder,
  onLevelClick,
  selectedSign, // <--- ADD THIS LINE
}: any) {
  const targetZ = useRef(0);
  const ROAD_PADDING = 12;

  const sortedSigns = useMemo(
    () =>
      [...signs].sort((a, b) => {
        const diffMap: any = { easy: 1, medium: 2, hard: 3 };
        return diffMap[a.difficulty] - diffMap[b.difficulty] || a.id - b.id;
      }),
    [signs],
  );
  const currentIndex = useMemo(() => {
    const idx = sortedSigns.findIndex((s) => !s.is_completed && !s.is_locked);
    return idx === -1 ? 0 : idx;
  }, [sortedSigns]);

  const totalLength = sortedSigns.length * 8;

  const isTierCompleted = (difficulty: string) => {
    const tierSigns = signs.filter((s: any) => s.difficulty === difficulty);
    return tierSigns.length > 0 && tierSigns.every((s: any) => s.is_completed);
  };

  const curve = useMemo(() => {
    const pts = [];
    for (let i = -ROAD_PADDING; i < sortedSigns.length + ROAD_PADDING; i++) {
      pts.push(new THREE.Vector3(Math.sin(i * 0.7) * 4.5, 0, -i * 8));
    }
    return new THREE.CatmullRomCurve3(pts);
  }, [sortedSigns.length]);

  const grassScatter = useMemo(() => {
    return Array.from({ length: 250 }).map((_, i) => ({
      x: (Math.random() - 0.5) * 150,
      z: Math.random() * (totalLength + ROAD_PADDING * 16) - ROAD_PADDING * 8,
      scale: 1 + Math.random() * 2,
      rotation: Math.random() * Math.PI,
    }));
  }, [totalLength]);

  return (
    <div className="w-full h-full bg-[#96d35f] cursor-grab active:cursor-grabbing touch-none overflow-hidden">
      <Canvas shadows dpr={[1, 2]}>
        <fog attach="fog" args={["#96d35f", 10, 90]} />
        <PerspectiveCamera makeDefault fov={30} position={[0, 20, 18]} />
        <WorldDragger totalLength={totalLength} targetZ={targetZ} />

        <Suspense
          fallback={
            <Html center className="text-white font-black">
              Building World...
            </Html>
          }
        >
          <Environment preset="forest" />
          <ambientLight intensity={0.7} />
          <directionalLight
            position={[15, 35, 15]}
            intensity={1.5}
            castShadow
            shadow-mapSize={[2048, 2048]}
          />

          <mesh
            rotation={[-Math.PI / 2, 0, 0]}
            position={[0, -0.6, -totalLength / 2]}
            receiveShadow
          >
            <planeGeometry args={[400, totalLength + 600]} />
            <meshStandardMaterial color="#8dc63f" />
          </mesh>

          <mesh position={[0, -0.55, 0]} scale={[1, 0.02, 1]} receiveShadow>
            <tubeGeometry
              args={[
                curve,
                (sortedSigns.length + ROAD_PADDING * 2) * 10,
                3.5,
                8,
                false,
              ]}
            />
            <meshStandardMaterial color="#a68d60" roughness={1} metalness={0} />
          </mesh>

          {grassScatter.map((g, i) => (
            <Grass
              key={`grass-${i}`}
              position={[g.x, -0.5, -g.z]}
              scale={g.scale}
              rotation={[0, g.rotation, 0]}
            />
          ))}

          {Array.from({ length: sortedSigns.length + ROAD_PADDING * 2 }).map(
            (_, idx) => {
              const i = idx - ROAD_PADDING;
              const z = -i * 8;
              const roadX = Math.sin(i * 0.7) * 4.5;

              return (
                <group key={i}>
                  <Rock
                    position={[roadX - 5 - Math.random(), -0.6, z + 2]}
                    scale={1.5}
                  />
                  <Rock
                    position={[roadX + 4 + Math.random(), -0.6, z - 1]}
                    scale={1.2}
                  />
                  <Tree position={[roadX - 25, -0.6, z + 2]} scale={2.5} />
                  <Tree position={[roadX + 28, -0.6, z - 5]} scale={3} />
                  <Tree position={[roadX - 7, -0.6, z + 1]} scale={1.2} />
                  <Tree position={[roadX + 7, -0.6, z - 2]} scale={1.1} />

                  {i >= 0 && i < sortedSigns.length && (
                    <>
                      <Rock
                        position={[
                          roadX + (Math.random() > 0.5 ? 10 : -10),
                          -0.6,
                          z + 5,
                        ]}
                        scale={5}
                      />
                      {i === 1 && (
                        <GltfModel
                          path="/world/Rock3.glb"
                          position={[roadX - 40, -5, z - 20]}
                          scale={18}
                          rotation={[0, 1, 0]}
                        />
                      )}
                      {i === Math.floor(sortedSigns.length / 2) && (
                        <GltfModel
                          path="/world/Rock3.glb"
                          position={[roadX + 45, -5, z - 20]}
                          scale={20}
                          rotation={[0, -0.5, 0]}
                        />
                      )}
                      <GltfModel
                        path={
                          Math.random() > 0.5
                            ? "/world/Bush.glb"
                            : "/world/Bush1.glb"
                        }
                        position={[roadX - 3.5, -0.5, z + 4]}
                        scale={1.5}
                      />
                      <GltfModel
                        path="/world/Bush.glb"
                        position={[roadX + 4.2, -0.5, z - 2]}
                        scale={1.2}
                      />

                      {i % 6 === 0 && (
                        <group position={[roadX + 15, -0.6, z - 4]}>
                          <GltfModel
                            path="/world/House1.glb"
                            scale={4}
                            rotation={[0, -0.5, 0]}
                            position={[0, 2, 0]}
                          />
                          <GltfModel
                            path="/world/House.glb"
                            position={[-6, 0, -2]}
                            scale={3.5}
                            rotation={[0, 0.2, 0]}
                          />
                          <GltfModel
                            path="/world/Fountain.glb"
                            position={[-3, 0, 6]}
                            scale={2}
                          />
                        </group>
                      )}
                      {i % 9 === 0 && (
                        <group position={[roadX - 18, -0.6, z - 10]}>
                          <GltfModel
                            path="/world/Market.glb"
                            scale={4}
                            rotation={[0, 0.4, 0]}
                          />
                          <GltfModel
                            path="/world/House.glb"
                            position={[7, 0, 3]}
                            scale={3.5}
                            rotation={[0, -0.3, 0]}
                          />
                        </group>
                      )}
                      {i % 14 === 0 && (
                        <GltfModel
                          path="/world/Pond.glb"
                          position={[roadX + 35, -0.55, z - 20]}
                          scale={5}
                        />
                      )}
                    </>
                  )}
                </group>
              );
            },
          )}

          {sortedSigns.map((sign, i) => {
            const z = -i * 8;
            const x = Math.sin(i * 0.7) * 4.5;
            const isSelected = selectedSign?.id === sign.id;
            const isCurrentProgress = i === currentIndex;

            return (
              <group key={sign.id} position={[x, 0, z]}>
                {/* 3D Ground Base */}
                <mesh position={[0, -0.45, 0]} receiveShadow>
                  <cylinderGeometry args={[1.2, 1.3, 0.2, 32]} />
                  <meshStandardMaterial color="#5d4037" roughness={1} />
                </mesh>

                {/* --- LAYER 1: THE NODE (On the ground) --- */}
                <Html
                  transform
                  sprite
                  distanceFactor={20}
                  position={[0, 0.5, 0]} // Attached to the base
                  zIndexRange={[0, 10]}
                >
                  <RoadmapNode
                    sign={sign}
                    index={i}
                    isCurrent={isCurrentProgress}
                    onClick={onLevelClick}
                  />
                </Html>

                {(isSelected || isCurrentProgress) && (
                  <Html
                    transform
                    sprite
                    distanceFactor={15}
                    position={[0, 7.5, 0]}
                    zIndexRange={[20, 100]}
                  >
                    <div className="flex flex-col items-center pointer-events-none">
                      <div className="relative pointer-events-auto">
                        {isSelected ? (
                          <SignPreviewCard
                            sign={sign}
                            onClose={() => onLevelClick(null)}
                          />
                        ) : isCurrentProgress ? (
                          <div className="flex flex-col items-center animate-float">
                            <MapPin
                              className="text-yellow-400 fill-yellow-100 mt-2"
                              size={36}
                            />
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </Html>
                )}

                {(() => {
                  const isLastOfTier =
                    i < sortedSigns.length - 1
                      ? sortedSigns[i].difficulty !==
                        sortedSigns[i + 1].difficulty
                      : true;

                  if (isLastOfTier) {
                    const qZ = z - 6;
                    const qX = x + (i % 2 === 0 ? 4 : -4); // Place lamp on the side of the road
                    const qSelected =
                      selectedSign?.id === `quiz-${sign.difficulty}`;
                    const qUnlocked = isTierCompleted(sign.difficulty);

                    return (
                      <group position={[qX - x, 0, qZ - z]}>
                        {/* 3D Stone Pedestal for the Lamp */}
                        <mesh position={[0, -0.45, 0]} receiveShadow>
                          <cylinderGeometry args={[1.5, 1.6, 0.2, 32]} />
                          <meshStandardMaterial
                            color="#475569"
                            roughness={0.8}
                          />
                        </mesh>

                        <QuizLamp
                          isLocked={!qUnlocked}
                          isCompleted={false}
                          onClick={() =>
                            onLevelClick({
                              id: `quiz-${sign.difficulty}`,
                              difficulty: sign.difficulty,
                              category: sign.category,
                            })
                          }
                        />

                        {/* THE POP-UP CARD (Stays as HTML) */}
                        <Html
                          transform
                          sprite
                          distanceFactor={25}
                          position={[0, 4, 0]}
                          zIndexRange={[20, 100]}
                        >
                          {qSelected && (
                            <QuizPreviewCard
                              difficulty={sign.difficulty}
                              category={sign.category}
                              onClose={() => onLevelClick(null)}
                            />
                          )}
                        </Html>
                      </group>
                    );
                  }
                })()}
              </group>
            );
          })}

          <ContactShadows opacity={0.4} scale={300} blur={2.5} far={80} />
          <BakeShadows />
        </Suspense>
      </Canvas>
    </div>
  );
}

useGLTF.preload([
  "/world/Tree.glb",
  "/world/Tree1.glb",
  "/world/Tree2.glb",
  "/world/Tree3.glb",
  "/world/Bush.glb",
  "/world/Bush1.glb",
  "/world/House.glb",
  "/world/House1.glb",
  "/world/Rock.glb",
  "/world/Rock1.glb",
  "/world/Rock2.glb",
  "/world/Rock3.glb",
  "/world/Pond.glb",
  "/world/Market.glb",
  "/world/Fountain.glb",
  "/world/grass.glb",
  "/world/grass2.glb",
]);
