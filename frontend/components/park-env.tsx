"use client";

import { useMemo } from "react";
import {
  useGLTF,
  Float,
  Cloud,
  Clouds,
  Sky,
  Sparkles,
} from "@react-three/drei";
import * as THREE from "three";

type GLTFResult = THREE.Object3D & {
  scene: THREE.Group;
};

function WorldProp({
  path,
  position,
  rotation = [0, 0, 0],
  scale = 1,
  randomRotate = true,
}: any) {
  const { scene } = useGLTF(path) as unknown as GLTFResult;

  const clonedScene = useMemo(() => {
    const clone = scene.clone();
    clone.traverse((child: any) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        if (child.material) {
          // 🔽 SOFTER MATERIAL (less shiny)
          child.material.roughness = 1;
          child.material.metalness = 0;
          child.material.envMapIntensity = 0.3;
        }
      }
    });
    return clone;
  }, [scene]);

  const finalRotation = useMemo(() => {
    if (!randomRotate) return rotation;
    return [rotation[0], rotation[1] + Math.random() * Math.PI, rotation[2]];
  }, [rotation, randomRotate]);

  return (
    <primitive
      object={clonedScene}
      position={position}
      rotation={finalRotation}
      scale={scale}
    />
  );
}

export function ParkEnv() {
  return (
    <group>
      {/* 1. LIGHTING (NEW - SOFT & BALANCED) */}
      <ambientLight intensity={0.4} />

      <directionalLight
        position={[20, 30, 10]}
        intensity={0.6}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />

      <hemisphereLight args={["#dbeafe", "#4d7c0f", 0.4]} />

      {/* 2. ATMOSPHERE & FOG */}
      <fog attach="fog" args={["#bae6fd", 15, 70]} />

      <Sky
        sunPosition={[50, 10, 50]} // 🔽 lower sun = softer light
        inclination={0.3}
        azimuth={0.25}
      />

      <Sparkles count={100} scale={20} size={1.5} speed={0.4} opacity={0.2} />

      {/* 3. CLOUDS */}
      <Clouds material={THREE.MeshBasicMaterial}>
        <Cloud
          seed={10}
          bounds={[20, 2, 20]}
          volume={10}
          color="white"
          opacity={0.3}
          position={[0, 20, -40]}
          speed={0.2}
        />
        <Cloud
          seed={20}
          bounds={[20, 2, 20]}
          volume={15}
          color="white"
          opacity={0.2}
          position={[30, 25, -50]}
          speed={0.2}
        />
      </Clouds>

      {/* 4. GROUND */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]} receiveShadow>
        <planeGeometry args={[300, 300]} />
        <meshStandardMaterial color="#5d8a33" roughness={1} />
      </mesh>

      {/* 5. PATHWAY */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -1.98, 0]}
        receiveShadow
      >
        <circleGeometry args={[10, 64]} />
        <meshStandardMaterial color="#e3d5a2" roughness={1} />
      </mesh>

      {/* 6. PROPS */}
      <WorldProp
        path="/world/Fountain.glb"
        position={[0, -2, -15]}
        scale={3.5}
        randomRotate={false}
      />
      <WorldProp
        path="/world/Bench.glb"
        position={[-9, 0, -3]}
        rotation={[0, 3, 0]}
        scale={8.2}
        randomRotate={false}
      />
      <WorldProp
        path="/world/PicnicTable.glb"
        position={[11, -2, -6]}
        rotation={[0, -0.5, 0]}
        scale={2}
        randomRotate={false}
      />

      {/* 7. TREES */}
      <WorldProp path="/world/Tree3.glb" position={[-22, -2, -15]} scale={3} />
      <WorldProp path="/world/Tree3.glb" position={[25, -2, -28]} scale={3} />
      <WorldProp path="/world/Tree3.glb" position={[-28, -2, -40]} scale={4} />
      <WorldProp path="/world/Tree3.glb" position={[-35, -2, -92]} scale={4} />
      <WorldProp path="/world/Tree3.glb" position={[42, -2, -92]} scale={4} />
      <WorldProp
        path="/world/Tree3.glb"
        position={[-10, -2, -55]}
        scale={4.5}
      />
      <WorldProp
        path="/world/Tree3.glb"
        position={[-45, -2, -25]}
        scale={3.5}
      />
      <WorldProp path="/world/Tree3.glb" position={[8, -2, -50]} scale={3.8} />
      <WorldProp path="/world/Tree3.glb" position={[35, -2, -35]} scale={4.2} />
      <WorldProp path="/world/Tree3.glb" position={[45, -2, -50]} scale={5} />
      <WorldProp path="/world/Tree3.glb" position={[20, -2, -55]} scale={4} />

      {/* 8. BUSHES */}
      <WorldProp path="/world/Bush.glb" position={[-12, -2, -8]} scale={2.5} />
      <WorldProp path="/world/Bush.glb" position={[10, -2, 2]} scale={2} />
      <WorldProp path="/world/Bush.glb" position={[-4, -2, -12]} scale={1.2} />
      <WorldProp path="/world/Bush1.glb" position={[14, -2, -4]} scale={2.2} />

      {/* 9. BASE GRASS */}
      {[
        [-5, 5],
        [6, -6],
        [-8, -10],
        [16, 5],
        [-12, 12],
        [0, 8],
        [-14, 2],
        [12, -12],
        [-3, -5],
        [5, 4],
        [9, -18],
        [-18, -5],
        [2, -10],
        [-25, -5],
        [25, 5],
        [-20, 15],
        [20, -20],
        [-10, 20],
        [15, 15],
        [-30, -10],
        [30, 10],
        [0, -25],
        [5, 25],
        [-5, -25],
        [10, 18],
        [-12, -18],
        [22, -8],
        [-22, 8],
        [-35, 0],
        [35, -5],
        [0, -35],
        [-15, -25],
        [15, -30],
        [-28, 20],
        [28, 25],
        [-40, -15],
        [40, 10],
        [-5, 40],
        [10, -40],
      ].map((pos, i) => (
        <WorldProp
          key={`base-${i}`}
          path={i % 2 === 0 ? "/world/grass.glb" : "/world/grass2.glb"}
          position={[pos[0], -2, pos[1]]}
          scale={1.1 + Math.random() * 0.7}
        />
      ))}

      {/* 10. EXTRA DENSE GRASS */}
      {Array.from({ length: 120 }).map((_, i) => {
        const x = (Math.random() - 0.5) * 100;
        const z = (Math.random() - 0.5) * 100;

        return (
          <WorldProp
            key={`extra-${i}`}
            path={i % 2 === 0 ? "/world/grass.glb" : "/world/grass2.glb"}
            position={[x, -2, z]}
            scale={0.8 + Math.random() * 0.8}
          />
        );
      })}
    </group>
  );
}

/* PRELOAD */
useGLTF.preload("/world/Tree3.glb");
useGLTF.preload("/world/Bush.glb");
useGLTF.preload("/world/Bush1.glb");
useGLTF.preload("/world/grass.glb");
useGLTF.preload("/world/grass2.glb");
useGLTF.preload("/world/Bench.glb");
useGLTF.preload("/world/Fountain.glb");
useGLTF.preload("/world/PicnicTable.glb");
