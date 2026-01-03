# 🦖 Digital-Fossil-GS

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Nerfstudio](https://img.shields.io/badge/Nerfstudio-Framework-green)](https://docs.nerf.studio/)

> **"My attempt at preserving NIT Rourkela's heritage using 3D AI."**

## 👋 About This Project
I am a 2nd-year AI student, and I wanted to explore how computers "see" in 3D. Everyone talks about 3D Gaussian Splatting being the next big thing after NeRFs, so I decided to build a project around it.

I recorded a video of a statue on my campus and wrote this pipeline to turn that 2D video into a fully explorable 3D scene.

---

## 🎥 The Result
*(Processed on Google Colab using a T4 GPU)*

![Demo GIF](assets/demo_orbit.gif)

---

## ⚙️ How I Built It
I used the **Nerfstudio** framework. Here is my understanding of what is happening under the hood:

1.  **Tracking the Camera (SfM):** First, I used a tool called COLMAP to figure out where my phone was in 3D space for every frame of the video.
2.  **The "Splatting":** Instead of building a mesh (like in video games), the model creates millions of tiny 3D ellipsoids (blobs).
3.  **Training:** The AI moves, stretches, and recolors these blobs until they look exactly like my video frames. It's like a 3D pointillism painting that learns by itself.

---

## 📊 My Experiment: Splatting vs. NeRF
I ran two different models on the same video to see the difference.

| Metric | Gaussian Splatting (My Project) | Traditional NeRF |
| :--- | :--- | :--- |
| **Training Time** | **~20 mins** (Fast) | ~45 mins (Slow) |
| **Visuals** | **Sharp edges** | A bit blurry |
| **FPS** | **120 FPS** (Real-time) | < 1 FPS |

**Verdict:** Gaussian Splatting is much better for real-time viewing.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Library:** Nerfstudio, PyTorch
* **Hardware:** Google Colab Free Tier
* **Visualization:** WandB

---

## 📝 What I Learned
* Setting up the environment and CUDA versions is the hardest part.
* "Structure from Motion" is crucial—if the video is shaky, the 3D model fails.
* How to read and interpret Loss Curves (PSNR) to know if the model is actually learning.

---
**Author:** Aryan Shukla
*Student @ NIT Rourkela*
