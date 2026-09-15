"use client";

const technologies = [
  {
    name: "YOLOv8",
    role: "Athlete Detection",
    description:
      "Localizes the primary athlete and extracts the relevant Region of Interest."
  },
  {
    name: "ResNet-50",
    role: "Spatial Feature Extraction",
    description:
      "Converts athlete regions into compact 2048-dimensional feature embeddings."
  },
  {
    name: "Bi-LSTM",
    role: "Temporal Modeling",
    description:
      "Learns forward and backward temporal dependencies across video frames."
  },
  {
    name: "Self-Attention",
    role: "Temporal Focus",
    description:
      "Assigns higher importance to frames that contain decisive action information."
  }
];

export default function ResultStats() {
  return (
    <section className="technology-section">
      <div className="section-heading">
        <p className="section-label">TECHNOLOGY</p>

        <h2>
          Built for <span>sports intelligence</span>
        </h2>
      </div>

      <div className="technology-grid">
        {technologies.map((technology) => (
          <div className="technology-card" key={technology.name}>
            <h3>{technology.name}</h3>

            <p className="technology-role">
              {technology.role}
            </p>

            <p>{technology.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}