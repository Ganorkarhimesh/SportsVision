"use client";

import { useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const supportedActions = [
  "Basketball",
  "Biking",
  "Diving",
  "Golf Swing",
  "Horse Riding",
  "Soccer Juggling",
  "Swing",
  "Tennis Swing",
  "Trampoline Jumping",
  "Volleyball Spiking",
  "Walking",
];

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoURL, setVideoURL] = useState("");

  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [topPredictions, setTopPredictions] = useState([]);
  const [testAccuracy, setTestAccuracy] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ============================================================
  // HANDLE VIDEO SELECTION
  // ============================================================

  function handleFileChange(event) {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);
    setVideoURL(URL.createObjectURL(file));

    // Reset previous results
    setPrediction(null);
    setConfidence(null);
    setTopPredictions([]);
    setTestAccuracy(null);
    setError("");
  }

  // ============================================================
  // ANALYZE VIDEO
  // ============================================================

  async function handleAnalyze() {
    if (!selectedFile) {
      setError("Please select a video first.");
      return;
    }

    setLoading(true);

    // Reset previous results
    setPrediction(null);
    setConfidence(null);
    setTopPredictions([]);
    setTestAccuracy(null);
    setError("");

    try {
      const formData = new FormData();

      formData.append("file", selectedFile);

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction failed."
        );
      }

      if (!data.success) {
        throw new Error(
          "The server could not analyze the video."
        );
      }

      // ========================================================
      // NEW BACKEND RESPONSE FORMAT
      // ========================================================

      const result = data.prediction;

      setPrediction(result.prediction);

      setConfidence(result.probability);

      setTopPredictions(
        result.top_predictions || []
      );

      setTestAccuracy(
        result.test_accuracy
      );

    } catch (err) {
      console.error(err);

      setError(
        err.message ||
        "Unable to connect to the SportsVision backend."
      );

    } finally {
      setLoading(false);
    }
  }

  // ============================================================
  // FORMAT PREDICTION NAME
  // ============================================================

  function formatPrediction(value) {
    if (!value) {
      return "";
    }

    return value
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) =>
        letter.toUpperCase()
      );
  }

  // ============================================================
  // FORMAT FILE SIZE
  // ============================================================

  function formatFileSize(bytes) {
    if (!bytes) {
      return "0 KB";
    }

    const mb = bytes / (1024 * 1024);

    if (mb >= 1) {
      return `${mb.toFixed(2)} MB`;
    }

    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  // ============================================================
  // UI
  // ============================================================

  return (
    <div className="page">

      {/* ======================================================
          NAVBAR
      ====================================================== */}

      <nav className="navbar">

        <div className="logo">

          <div className="logo-mark">
            SV
          </div>

          <span>
            SportsVision
          </span>

        </div>

        <div className="nav-links">

          <a href="#analyzer">
            Analyzer
          </a>

          <a href="#pipeline">
            Pipeline
          </a>

          <a href="#actions">
            Actions
          </a>

        </div>

      </nav>


      <main className="container">

        {/* ====================================================
            PAGE HEADER
        ==================================================== */}

        <section className="page-header">

          <h1>
            Sports Action Recognition
          </h1>

          <p>
            Upload a sports video and let the SportsVision
            computer vision pipeline identify the action
            being performed.
          </p>

        </section>


        {/* ====================================================
            ANALYZER
        ==================================================== */}

        <section
          className="analyzer"
          id="analyzer"
        >

          {/* ==================================================
              VIDEO PANEL
          ================================================== */}

          <div className="panel video-panel">

            <div className="panel-header">

              <h2>
                Video Input
              </h2>

              <span>
                MP4 · AVI · MOV · MKV · MPG
              </span>

            </div>


            <div className="video-area">

              {videoURL ? (

                <video
                  className="video-preview"
                  src={videoURL}
                  controls
                />

              ) : (

                <div className="upload-area">

                  <div className="upload-content">

                    <div className="upload-icon">
                      ↑
                    </div>

                    <h3>
                      Select a sports video
                    </h3>

                    <p>
                      Upload a video file to begin analysis
                    </p>

                    <label
                      htmlFor="video-upload"
                      className="browse-button"
                    >
                      Browse Video
                    </label>

                    <input
                      id="video-upload"
                      className="file-input"
                      type="file"
                      accept=".mp4,.avi,.mov,.mkv,.mpg,video/*"
                      onChange={handleFileChange}
                    />

                  </div>

                </div>

              )}

            </div>


            {/* =================================================
                VIDEO FOOTER
            ================================================= */}

            <div className="video-footer">

              <div className="file-details">

                {selectedFile ? (

                  <>
                    <div className="file-name">
                      {selectedFile.name}
                    </div>

                    <div className="file-info">
                      {formatFileSize(
                        selectedFile.size
                      )}
                    </div>
                  </>

                ) : (

                  <>
                    <div className="file-name">
                      No video selected
                    </div>

                    <div className="file-info">
                      Select a video to start
                    </div>
                  </>

                )}

              </div>


              <button
                className="analyze-button"
                onClick={handleAnalyze}
                disabled={
                  !selectedFile ||
                  loading
                }
              >

                {loading
                  ? "Analyzing..."
                  : "Analyze Video"}

              </button>

            </div>

          </div>


          {/* ==================================================
              RESULT PANEL
          ================================================== */}

          <div className="panel result-panel">

            <div className="panel-header">

              <h2>
                Analysis Result
              </h2>

              <span>
                Model Output
              </span>

            </div>


            {/* =================================================
                LOADING
            ================================================= */}

            {loading ? (

              <div className="empty-result">

                <div className="loading">

                  <div className="loading-spinner"></div>

                  <p>
                    Processing video...
                  </p>

                </div>

              </div>


            ) : prediction ? (

              /* =================================================
                 RESULT AVAILABLE
              ================================================= */

              <div className="result-content">

                {/* ---------------------------------------------
                    DETECTED ACTION
                --------------------------------------------- */}

                <div className="result-label">
                  Detected Action
                </div>

                <div className="prediction">
                  {formatPrediction(
                    prediction
                  )}
                </div>


                {/* ---------------------------------------------
                    PREDICTION PROBABILITY
                --------------------------------------------- */}

                <div className="confidence-header">

                  <span>
                    Prediction probability
                  </span>

                  <span className="confidence-value">
                    {confidence.toFixed(2)}%
                  </span>

                </div>


                <div className="confidence-bar">

                  <div
                    className="confidence-fill"
                    style={{
                      width: `${Math.min(
                        confidence,
                        100
                      )}%`,
                    }}
                  ></div>

                </div>


                <div className="result-divider"></div>


                {/* =================================================
                    TOP 3 PREDICTIONS
                ================================================= */}

                <div className="top-predictions">

                  <div className="result-label">
                    Top Predictions
                  </div>


                  {topPredictions.map(
                    (item, index) => (

                      <div
                        className="prediction-row"
                        key={item.class}
                      >

                        <div className="prediction-rank">
                          {index + 1}
                        </div>


                        <div className="prediction-name">
                          {formatPrediction(
                            item.class
                          )}
                        </div>


                        <div className="prediction-probability">
                          {item.probability.toFixed(2)}%
                        </div>

                      </div>

                    )
                  )}

                </div>


                <div className="result-divider"></div>


                {/* =================================================
                    MODEL STATISTICS
                ================================================= */}

                <div className="result-stats">

                  <div className="stat">

                    <span className="stat-value">
                      32
                    </span>

                    <span className="stat-label">
                      Frames
                    </span>

                  </div>


                  <div className="stat">

                    <span className="stat-value">
                      2048
                    </span>

                    <span className="stat-label">
                      Features
                    </span>

                  </div>


                  <div className="stat">

                    <span className="stat-value">
                      {testAccuracy
                        ? `${testAccuracy.toFixed(2)}%`
                        : "--"}
                    </span>

                    <span className="stat-label">
                      Test Accuracy
                    </span>

                  </div>

                </div>


                {/* =================================================
                    ERROR
                ================================================= */}

                {error && (

                  <div className="error-message">
                    {error}
                  </div>

                )}

              </div>


            ) : (

              /* =================================================
                 NO RESULT
              ================================================= */

              <div className="empty-result">

                <div>

                  <div className="empty-result-icon">
                    ○
                  </div>

                  <h3>
                    No analysis available
                  </h3>

                  <p>
                    Upload a video and click Analyze
                    Video to see the prediction.
                  </p>

                </div>

              </div>

            )}


            {/* =================================================
                ERROR WITHOUT RESULT
            ================================================= */}

            {error &&
              !prediction &&
              !loading && (

                <div
                  style={{
                    padding:
                      "0 25px 20px",
                  }}
                >

                  <div className="error-message">
                    {error}
                  </div>

                </div>

              )}

          </div>

        </section>


        {/* ====================================================
            MODEL PIPELINE
        ==================================================== */}

        <section
          className="section"
          id="pipeline"
        >

          <div className="section-title">

            <h2>
              Model Pipeline
            </h2>

            <p>
              The video passes through multiple stages
              before the final action classification.
            </p>

          </div>


          <div className="pipeline">

            <div className="pipeline-step">

              <div className="step-number">
                01
              </div>

              <h3>
                YOLOv8
              </h3>

              <p>
                Detects and localizes the athlete
                in each frame.
              </p>

            </div>


            <div className="pipeline-step">

              <div className="step-number">
                02
              </div>

              <h3>
                ResNet-50
              </h3>

              <p>
                Extracts 2048-dimensional spatial
                visual features.
              </p>

            </div>


            <div className="pipeline-step">

              <div className="step-number">
                03
              </div>

              <h3>
                Bi-LSTM
              </h3>

              <p>
                Learns temporal relationships across
                video frames.
              </p>

            </div>


            <div className="pipeline-step">

              <div className="step-number">
                04
              </div>

              <h3>
                Self-Attention
              </h3>

              <p>
                Focuses on important moments in
                the action sequence.
              </p>

            </div>


            <div className="pipeline-step">

              <div className="step-number">
                05
              </div>

              <h3>
                Classifier
              </h3>

              <p>
                Predicts the sports action from
                11 supported classes.
              </p>

            </div>

          </div>

        </section>


        {/* ====================================================
            SUPPORTED ACTIONS
        ==================================================== */}

        <section
          className="section"
          id="actions"
        >

          <div className="section-title">

            <h2>
              Supported Actions
            </h2>

            <p>
              SportsVision is trained on the UCF11 dataset.
            </p>

          </div>


          <div className="actions-grid">

            {supportedActions.map(
              (action) => (

                <div
                  className="action"
                  key={action}
                >
                  {action}
                </div>

              )
            )}

          </div>

        </section>


        {/* ====================================================
            TECHNICAL DETAILS
        ==================================================== */}

        <section className="section">

          <div className="section-title">

            <h2>
              Technical Configuration
            </h2>

            <p>
              Core components used by the SportsVision system.
            </p>

          </div>


          <div className="tech-grid">

            <div className="tech-card">

              <h3>
                Input Sequence
              </h3>

              <p>
                32 sampled frames are processed
                from each input video.
              </p>

            </div>


            <div className="tech-card">

              <h3>
                Feature Representation
              </h3>

              <p>
                Each frame is represented using a
                2048-dimensional ResNet-50 feature vector.
              </p>

            </div>


            <div className="tech-card">

              <h3>
                Action Classes
              </h3>

              <p>
                The classifier predicts one of
                11 sports action categories.
              </p>

            </div>

          </div>

        </section>

      </main>


      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer className="footer">

        <div className="container">
          SportsVision · Human Action Recognition in Sports Videos
        </div>

      </footer>

    </div>
  );
}