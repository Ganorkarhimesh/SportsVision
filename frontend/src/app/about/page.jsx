import Navbar from "@/components/Navbar";

export default function AboutPage() {
  return (
    <>
      <Navbar />

      <main className="about-page">
        <section className="about-hero">
          <p className="section-label">ABOUT SPORTSVISION</p>

          <h1>
            Understanding Human Actions
            <span> in Sports Videos</span>
          </h1>

          <p>
            SportsVision is a hybrid deep learning system designed to recognize
            human actions in sports videos by combining athlete detection,
            spatial feature extraction, temporal modeling and attention.
          </p>
        </section>

        <section className="about-grid">
          <div className="about-card">
            <h2>01 — YOLOv8</h2>
            <p>
              YOLOv8 detects and localizes athletes in each sampled video frame.
              The detected athlete region is extracted as the Region of
              Interest.
            </p>
          </div>

          <div className="about-card">
            <h2>02 — ResNet-50</h2>
            <p>
              ResNet-50 extracts spatial features from the athlete ROI.
              Global average pooling produces a 2048-dimensional feature
              representation.
            </p>
          </div>

          <div className="about-card">
            <h2>03 — Bi-LSTM</h2>
            <p>
              A two-layer Bidirectional LSTM learns temporal dependencies from
              the sequence of spatial features.
            </p>
          </div>

          <div className="about-card">
            <h2>04 — Self-Attention</h2>
            <p>
              Temporal self-attention assigns higher importance to decisive
              frames that contribute more strongly to the final action.
            </p>
          </div>
        </section>

        <section className="methodology">
          <p className="section-label">METHODOLOGY</p>

          <div className="method-flow">
            <span>Sports Video</span>
            <span>→</span>
            <span>YOLOv8</span>
            <span>→</span>
            <span>ResNet-50</span>
            <span>→</span>
            <span>Bi-LSTM</span>
            <span>→</span>
            <span>Attention</span>
            <span>→</span>
            <span>Action</span>
          </div>
        </section>
      </main>
    </>
  );
}