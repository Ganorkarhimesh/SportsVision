"use client";

import { ArrowDown, Sparkles } from "lucide-react";

export default function Hero() {
  const scrollToUpload = () => {
    document.getElementById("upload")?.scrollIntoView({
      behavior: "smooth"
    });
  };

  return (
    <section className="hero">
      <div className="hero-content">
        <div className="hero-badge">
          <Sparkles size={16} />
          HYBRID DEEP LEARNING
        </div>

        <h1>
          Understand
          <br />
          <span>Human Actions</span>
          <br />
          in Sports Videos.
        </h1>

        <p>
          SportsVision combines YOLOv8, ResNet-50, Bi-LSTM and temporal
          self-attention to recognize human actions from sports videos.
        </p>

        <button className="primary-button" onClick={scrollToUpload}>
          Analyze Sports Video
          <ArrowDown size={18} />
        </button>
      </div>
    </section>
  );
}