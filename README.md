[comment]: <# ASW-planning>
# ASW Planning with Public Belief States and Self-Play

[comment]: <This repository contains the code used in the research paper *"Anti-Submarine Warfare Planning Using Public Belief States and Self-Play"* by FOI and KTH.>

This repository contains the code and experimental results for the paper "Anti-Submarine Warfare Planning Using Public Belief States and Self-Play" (ICMLA 2025).

See the full paper here: [Anti-Submarine Warfare Planning Using Public Belief States and Self-Play (FOI Report)](https://www.foi.se/download/18.2f6a97f619b2374cd1563/1766415729011/Anti-submarine_warfare_planning_FOI-S--7153--SE.pdf)


## Motivation

Modern anti-submarine warfare (ASW) requires unpredictable search patterns to counter stealthy underwater vehicles (UVs) that can observe and adapt to static or predictable search behavior. Traditional planning tools optimize search routes against fixed enemy models — making them exploitable. This work pioneers the application of *superhuman poker AI techniques* — specifically, public belief states and self-play reinforcement learning — to ASW, enabling truly adversarial, game-theoretically optimal search strategies.

## Scenario: The Flaming Datum Problem

We model ASW as a two-player, imperfect-information game on a hexagonal grid (see illustration below):

- **Player 1 (Intruder)**: A stealthy UV attempting to reach one of two critical infrastructure assets.
- **Player 2 (Defender)**: A patrol vessel with a dipping sonar, searching unpredictably to detect the UV.

The UV is assumed to know the position and activity of all active sonars in real time — a critical realism often ignored in prior work. The defender must act without knowing the UV’s location, updating its belief over time based on sonar reports.

The goal is to generate search paths that approximate a Nash equilibrium: unpredictable even to an adversary who knows the strategy.

![Hexagonal Search Grid](figures/Board2.png)  
*Illustration: Discretized nautical chart with axial coordinates (q,r). Land (masked) and water hexagons define the operational area. Critical assets are marked as flags.*

## Key Innovation

Unlike prior game-theoretic patrolling systems (e.g., PROTECT, PAWS) that assume passive sensors, this approach explicitly models *active sonars* — whose emissions reveal the defender’s position — making the problem far more dynamic and realistic. The solution transforms the imperfect-information ASW game into a perfect-information public belief game, solvable via self-play and regularization.

The code implements this method in JAX, enabling high-throughput GPU-accelerated training and real-time strategy generation.


[comment]: <# Index>

