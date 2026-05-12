# Physio-Rider: Gamified Upper-Limb Rehabilitation & AI Co-Driver Ecosystem

**Physio-Rider** transforms traditional hand-gesture-controlled 3D racing into an interactive, clinically motivated upper-limb physical therapy platform. Designed to eliminate the repetition and boredom of standard rehabilitation regimens, the software leverages real-time AI computer vision and native conversational companions to gamify shoulder elevation, wrist supination/pronation, and proper spinal posture.

---

## 🌟 Clinical Vision & Motivation
Physical therapy for upper-extremity recovery (such as post-stroke motor relearning, rotator cuff rebuilding, and frozen shoulder rehabilitation) often suffers from low adherence due to monotony and fatigue. **Physio-Rider** reimagines these exercises as an immersive driving experience where the patient's physical range of motion directly drives the target vehicle's capabilities. 

By requiring continuous arm elevation to maintain acceleration and rewarding smooth, controlled steering motions, patients experience immediate, positive gamified reinforcement for achieving their physical therapy goals.

---

## 🚀 Key Features & Subsystems

### 1. High-Fidelity Glassmorphism Telemetry Interface
A beautiful, clinical-grade real-time heads-up display (HUD) overlaying the 3D world provides instantaneous visual feedback to both the patient and supervising therapists:
- **Arm Elevation Gauge:** Translates absolute vertical hand coordinates into a mapped `0% - 100%` therapeutic threshold. Dynamic color states (Standard → High → Optimal) incentivize reaching recommended elevation ranges.
- **Sustained Hold Timer:** Accumulates continuous driving seconds maintained above base elevation goals to build muscular endurance.
- **Steering Smoothness Score:** Computes a moving-average filter over steering angles to penalize erratic jerkiness and heavily reward fluid, controlled wrist movements.
- **Spinal/Neck Alignment Monitor:** Tracks vertical facial axis ratios to detect backward neck extension or forward slouching, instructing active relaxation.

### 2. Proactive AI Conversational Buddy (`PhysioBuddy`)
To ensure the patient never feels isolated during their session, the application implements an emotionally intelligent co-driver companion natively inside the browser:
- **Continuous Facial Analysis:** Interprets video feeds to ensure camera line-of-sight is maintained. If tracking is lost or looking away occurs for multiple seconds, the companion proactively speaks out loud (*"Hey partner! I lost sight of your face..."*).
- **Distance Milestone Interactions:** Triggers interactive multi-turn dialogue loops at predefined driving targets (`60m`, `150m`), connecting physical exertion to daily dietary protein/hydration discussions and recommending post-rehab wall routines.
- **Native Player Voice Recognition:** Employs built-in Speech-to-Text inference to actively listen to the patient's voice after questions, auto-transcribing inputs (`🎤 You: "..."`) and matching intent keywords hands-free.

### 3. Therapeutic Physics Mapping
- **Rehab Drive Constraints:** The vehicle accelerates forward only when both hands are open and elevated beyond base resting bounds. Closing hands into fists acts as an instant emergency brake.
- **Proportional Speed Boost:** Maximum cruising speed dynamically scales up to **1.5x velocity** based on absolute arm lift height, connecting effort directly to in-game rewards.

---

## 🛠️ Technical Stack & Client Architecture
- **Rendering Engine:** Vanilla Three.js rendering highly optimized low-poly procedurally populated city environments, responsive traffic bounding boxes, and ambient bird particle arrays.
- **Computer Vision Inference:** MediaPipe Tasks Vision Web API (`vision_bundle.mjs`) executing offline WebAssembly/WebGL accelerated hand and facial landmark extraction at 60fps.
- **Offline Natural Language:** Utilizes standard client-side `window.speechSynthesis` and `webkitSpeechRecognition` APIs, guaranteeing zero latency, offline readiness, and zero reliance on paid cloud inference keys.

---

## 📦 Setup & Deployment Guide

The web platform is entirely statically deployable and ready for rapid local setup in clinical spaces or remote home-care tablets.

### Prerequisites
- Modern web browser with microphone and camera permission capabilities (Google Chrome or Microsoft Edge highly recommended for native speech engine compatibility).
- Local web server utility to handle ES module cross-origin requirements.

### Running Locally
1. Clone the project repository:
   ```powershell
   git clone https://github.com/hasenalbanna/physio-rider.git
   cd physio-rider
   ```
2. Launch a lightweight static development server targeting the root directory:
   ```powershell
   npx http-server ./public -p 8080
   ```
3. Open your browser and navigate to:
   ```text
   http://127.0.0.1:8080
   ```
4. **Grant Webcam and Microphone Access** when prompted to enable live 3D physics rendering and active voice companionship loops.

---

## 🤝 Contribution & Extension
Custom sensitivity calibration profiles, multi-patient logging schemas, and custom driving terrain levels can be appended directly via modifying configuration globals inside `public/index.html`.
