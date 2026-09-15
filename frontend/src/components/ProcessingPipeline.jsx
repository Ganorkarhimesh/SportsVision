"use client";

import {
  ScanFace,
  Brain,
  Activity,
  Sparkles,
  Target
} from "lucide-react";

const steps = [
  {
    number: "01",
    title: "YOLOv8",
    description: "Athlete Detection",
    icon: ScanFace
  },
  {
    number: "02",
    title: "ResNet-50",
    description: "Spatial Features",
    icon: Brain
  },
  {
    number: "03",
    title: "Bi-LSTM",
    description: "Temporal Modeling",
    icon: Activity
  },
  {
    number: "04",
    title: "Attention",
    description: "Important Frames",
    icon: Sparkles
  },
  {
    number: "05",
    title: "Softmax",
    description: "Action Prediction",
    icon: Target
  }
];

export default function ProcessingPipeline() {
  return (
    <section className="pipeline-section">
      <div className="section-heading">
        <p className="section-label">HOW IT WORKS</p>

        <h2>
          From <span>video</span> to action
        </h2>

        <p>
          A hybrid deep learning pipeline combines spatial and temporal
          information to understand sports actions.
        </p>
      </div>

      <div className="pipeline">
        {steps.map((step, index) => {
          const Icon = step.icon;

          return (
            <div className="pipeline-item" key={step.number}>
              <div className="pipeline-card">
                <div className="pipeline-number">{step.number}</div>

                <Icon size={30} />

                <h3>{step.title}</h3>

                <p>{step.description}</p>
              </div>

              {index < steps.length - 1 && (
                <div className="pipeline-arrow">→</div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}